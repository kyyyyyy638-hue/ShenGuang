import 'dotenv/config';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { executeCommandTool, listDirectoryTool, readFileTool, writeFileTool } from './all-tools.mjs';

const model = new ChatOpenAI({
  // 原来硬编码的 "qwen-plus" 是阿里云百炼的模型名，智谱不认识，会报 Model not exist。
  // 改成从 .env 读，和项目里其他脚本保持一致
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  // GLM-5.3 强制开启思考模式：不显式传 thinking.type=enabled 请求会直接失败；
  // reasoning_effort 默认 max，demo 用 low 省 token
  modelKwargs: {
    thinking: { type: 'enabled' },
    reasoning_effort: process.env.REASONING_EFFORT || 'low',
  },
});

const tools = [readFileTool, writeFileTool, executeCommandTool, listDirectoryTool];

// 绑定工具到模型 —— 工具的名字/描述/参数格式会随每次请求发给模型，
// 所以 SystemMessage 里不需要再手写一遍工具清单（写了只是冗余，不是机制）
const modelWithTools = model.bindTools(tools);

// 平台信息注入 —— 模型不知道平台就会按 Unix 习惯编命令（ls / mkdir -p），在 Windows 全部失效
const IS_WIN = process.platform === 'win32';
const platformLine = IS_WIN
  ? '运行环境：Windows（shell 是 cmd.exe）。用 dir 而不是 ls，用 type 而不是 cat，用 findstr 而不是 grep。'
  : '运行环境：Unix（shell 是 /bin/sh）。';

// Agent 执行函数
async function runAgentWithTools(query, maxIterations = 30) {
  const messages = [
    new SystemMessage(`你是一个项目管理助手，使用工具完成任务。

${platformLine}

当前工作目录: ${process.cwd()}

重要规则 - execute_command：
- workingDirectory 参数会自动切换到指定目录
- 当使用 workingDirectory 时，绝对不要在 command 中使用 cd
- 错误示例: { command: "cd react-todo-app && pnpm install", workingDirectory: "react-todo-app" }
- 正确示例: { command: "pnpm install", workingDirectory: "react-todo-app" }
- 启动常驻服务（如 pnpm run dev）必须传 background: true，否则会一直阻塞到 120 秒超时
- 不要用 "start /b xxx"、"xxx &" 这类写法自己后台化，直接用 background: true

拿到工具结果后基于真实内容行动，不要凭空假设。回复要简洁，只说做了什么`),
    new HumanMessage(query),
  ];

  for (let i = 0; i < maxIterations; i++) {
    console.log(`⏳ 正在等待 AI 思考... (${i + 1}/${maxIterations})`);
    const response = await modelWithTools.invoke(messages);
    // 每一轮的模型回复都要进上下文（教程只在第一轮 push 的问题在这里不存在，
    // 因为 push 放在了循环体内部，每轮都会记录）
    messages.push(response);

    // 模型不再调工具 = 它决定用自然语言收尾，循环结束
    if (!response.tool_calls || response.tool_calls.length === 0) {
      console.log(`\n✨ AI 最终回复:\n${response.content}\n`);
      return response.content;
    }

    // 执行本轮所有工具调用
    for (const toolCall of response.tool_calls) {
      const foundTool = tools.find((t) => t.name === toolCall.name);

      let toolResult;
      if (!foundTool) {
        // 不补这条的话，这一轮就没有对应的 ToolMessage，
        // 下一轮模型会因为「tool_calls 没有得到响应」直接报协议错误
        toolResult = `错误: 找不到名为 "${toolCall.name}" 的工具。可用工具: ${tools.map((t) => t.name).join(', ')}`;
      } else {
        try {
          toolResult = await foundTool.invoke(toolCall.args);
        } catch (error) {
          // 工具抛错不能炸掉整个 agent —— 把错误当成结果喂回去，模型会自我修正
          toolResult = `工具执行出错: ${error.message}`;
        }
      }

      messages.push(
        new ToolMessage({
          content: toolResult,
          tool_call_id: toolCall.id,
        })
      );
    }
  }

  console.log(`\n⚠️ 达到最大迭代次数 ${maxIterations}，强制停止。`);
  return messages[messages.length - 1].content;
}

// ── 入口 ─────────────────────────────────────────────────────
// 用法：
//   node ./src/min-cursor.mjs                        （用默认任务）
//   node ./src/min-cursor.mjs 帮我创建一个 hello.txt 内容是 hi
const query = process.argv.slice(2).join(' ') || `完善 react-todo-app 项目里的 TodoList 应用。

项目现状：react-todo-app 目录下已经有一个 Vite + React + TypeScript 项目，依赖已安装。
不要重新创建脚手架（不要跑 pnpm create vite），不要用 npm，只用 pnpm。
所有命令都用 workingDirectory 参数指定目录，不要在 command 里写 cd。

一、先看清楚现状
用 list_directory 看 react-todo-app/src，再用 read_file 读现有的 App.tsx，然后才动手改。
不要凭猜测直接覆盖。

二、重写 src/App.tsx，实现功能完整的 TodoList
（即使现有代码已有一部分，也按下面的要求重写完整版本）
- 添加任务、删除任务、编辑任务（点编辑按钮进入编辑态，可保存或取消）、标记完成
- 分类筛选：全部 / 进行中 / 已完成
- 统计信息：总数、进行中数量、已完成数量
- localStorage 持久化：刷新页面后数据不丢

三、样式写在 src/App.css
- 蓝到紫的渐变背景
- 卡片阴影、圆角
- 鼠标悬停（hover）效果

四、动画用 CSS transition（不要引入任何动画库）
- 添加 / 删除时的过渡动画
- 完成状态切换的过渡

五、验收标准（逐条给出工具输出作为证据；拿不到证据的项就写"未验证"，不要凭印象宣布成功）
1. 运行 pnpm build（workingDirectory 传 react-todo-app），必须 exit code 为 0 才算通过。
   如果报错，读 stderr 定位并修，然后重新 build，直到 exit code 为 0。
   如果报依赖缺失，先跑 pnpm install（不要用 npm，本项目是 pnpm 的 hoisted 布局）。
2. 用 list_directory 列出 react-todo-app/src，确认 App.tsx 和 App.css 存在。
3. 用 pnpm run dev 启动 dev server 时必须传 background: true（前台执行会阻塞到 120 秒超时），
   然后跑 netstat -ano | findstr :5173 确认端口在监听。若 5173 已被占用，vite 会自动换端口，
   以命令输出里的实际端口为准。
   不要用 "start /b xxx"、"xxx &" 这类写法自己后台化，那样拿不到输出。

最后报告：每条验收标准对应给出工具返回的证据。
`;


try {
  await runAgentWithTools(query);
} catch (error) {
  console.error(`\n❌ 错误: ${error.message}\n`);
}
