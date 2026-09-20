// 完整的 agent 循环 —— 这就是 md2 的终点，也是 md3「mini cursor」的起点
// 运行：node ./src/tool-file-read-loop.mjs
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { readFileTool } from './tools/read-file.mjs';

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  // 去掉了 temperature：GLM-5.3 是强制思考的推理模型，官方示例不带该参数。
  // 将来若换回非推理模型，再把它加回来（tool calling 建议 temperature: 0）。
  //
  // GLM-5.3 必须显式传 thinking.type=enabled，否则请求直接失败（官方迁移提示）。
  modelKwargs: {
    thinking: { type: 'enabled' },
    reasoning_effort: process.env.REASONING_EFFORT || 'low',
  },
});

const tools = [readFileTool];
const modelWithTools = model.bindTools(tools);

const messages = [
  new SystemMessage(`你是一个代码助手，可以使用工具读取文件并解释代码。

工作流程：
1. 用户要求读取文件时，立即调用 read_file 工具
2. 等待工具返回文件内容
3. 基于文件内容进行分析和解释

可用工具：
- read_file: 读取文件内容（使用此工具来获取文件内容）
`),
  new HumanMessage('请读取 src/tools/read-file.mjs 文件内容并解释代码'),
];

let response = await modelWithTools.invoke(messages);

// ★ 关键 1：必须把模型这条「我要调工具」的回复也放进上下文
//    不放，下一轮它就不承认自己调用过 read_file，ToolMessage 会变成孤儿
messages.push(response);

while (response.tool_calls && response.tool_calls.length > 0) {
  console.log(`\n[检测到 ${response.tool_calls.length} 个工具调用]`);

  // 执行本轮所有工具调用
  const toolResults = await Promise.all(
    response.tool_calls.map(async (toolCall) => {
      const tool = tools.find((t) => t.name === toolCall.name);
      if (!tool) {
        return `错误: 找不到工具 ${toolCall.name}`;
      }

      console.log(`  [执行工具] ${toolCall.name}(${JSON.stringify(toolCall.args)})`);
      try {
        return await tool.invoke(toolCall.args);
      } catch (error) {
        return `错误: ${error.message}`;
      }
    })
  );

  // ★ 关键 2：用 tool_call_id 把「哪次调用」和「哪个结果」对上号
  response.tool_calls.forEach((toolCall, index) => {
    messages.push(
      new ToolMessage({
        content: toolResults[index],
        tool_call_id: toolCall.id,
      })
    );
  });

  // 带着工具结果再问一次模型
  response = await modelWithTools.invoke(messages);

  // ★ 关键 3（教程漏了）：新回复也要进上下文
  //    不加这行，只调一次工具时看不出问题；一旦模型需要连续调 2 个以上工具就会报
  //    「ToolMessage 找不到对应的 tool_calls」，因为上一轮的 AIMessage 没被记录
  messages.push(response);
}

console.log('\n[最终回复]');
console.log(response.content);
