// 对应教程正文里那份「不循环」的代码 —— 目的是先看清楚模型返回了什么
// 运行：node ./src/tool-file-read.mjs
import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
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

const response = await modelWithTools.invoke(messages);

console.log('===== 模型的完整返回 =====');
console.log(response);

console.log('\n===== tool_calls 单独看一眼 =====');
console.log(JSON.stringify(response.tool_calls, null, 2));
