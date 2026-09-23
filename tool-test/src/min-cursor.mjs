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
const query = process.argv.slice(2).join(' ') || `创建一个功能丰富的 React 天气卡片应用：
1. 创建项目：echo -e "n\nn" | pnpm create vite react-weather-card --template react-ts
2. 修改 src/App.tsx，实现完整功能的天气卡片：
- 搜索城市（输入框 + 搜索按钮，回车也能触发）、展示当前天气、切换展示城市、删除城市
- 当前天气信息：城市名、温度、天气状况文字、天气图标、体感温度、湿度、风速
- 单位切换：摄氏度 / 华氏度，切换后所有温度同步换算（演示：只做前端换算，不重新请求接口）
- 分类筛选不适用，改为：城市列表 / 当前展示城市 的区分高亮
- 统计信息显示：已保存城市数量、当前城市、数据更新时间
- 加载态与错误态：请求中显示 loading，失败显示明确错误信息 + 重试按钮，不要把错误吞掉假装成功
- localStorage 数据持久化：城市列表、当前选中城市、温度单位，刷新页面后不丢
- 数据源用 Open-Meteo（免费、无需 API Key）：
  地理编码 https://geocoding-api.open-meteo.com/v1/search?name={城市}&count=5&language=zh
  天气数据 https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code
  用 fetch 直接调，不要引入 axios 等额外请求库。
  weather_code 到中文天气文字 + 图标的映射表自己写，覆盖 0/1/2/3/45/48/51/53/55/61/63/65/71/73/75/77/80/81/82/85/86/95/96/99。
3. 添加复杂样式：
- 渐变背景（深蓝到青紫，呼应天气主题）
- 卡片毛玻璃效果（backdrop-filter: blur）、阴影、圆角
- 悬停效果：卡片上浮 + 阴影加深
- 响应式：窄屏（< 600px）下城市列表改成纵向堆叠
4. 添加动画：
- 添加/删除城市时的过渡动画（入场用 @keyframes，移除用 transition）
- 温度单位切换、数据刷新时数字的过渡（CSS transition，至少 color 或 transform）
- 卡片 hover 上浮也用 transition
- loading 转圈用 CSS @keyframes
- 不要引入任何动画库（禁止 framer-motion / react-spring / gsap）
5. 列出目录确认

注意：使用 pnpm，功能要完整，样式要美观，要有动画效果。
另外必须改 index.html 的 <title>，Vite 默认模板标题会导致"页面看起来没做完"。

之后在 react-weather-card 项目中：
1. 使用 pnpm install 安装依赖
2. 使用 pnpm run dev 启动服务器（必须传 background: true，前台执行会阻塞到 120 秒超时）
3. 用 netstat -ano | findstr :5173 确认端口在监听；若 5173 被占用，vite 会自动换端口，以实际输出为准
4. 【必做】运行时证据：curl 实际端口拿到首页 HTML，确认 <div id="root"> 存在且 <title> 已是天气卡片（不是 Vite 默认标题）
5. 【必做】数据链路证据：单独 curl 一次 Open-Meteo 接口，贴出返回的 JSON 片段。
   - 拿到有效数据 → 说明接口链路通。
   - 拿不到（沙箱/网络限制）→ 必须明确写出"外部 API 在当前环境不可达，天气数据端到端渲染未验证"，
     并读代码把错误处理分支贴出来证明降级逻辑真实存在。
   - 严禁因为 build 通过就宣布"天气功能正常"。
6. 读 package.json 的 dependencies 贴出内容，确认没有动画库和额外请求库

最后报告：每条验收标准对应给出工具返回的证据。
不允许出现"应该可以了""看起来没问题"这类表述——
只有 exit code、命令原始输出、文件内容可以当作证据，拿不到就写"未验证"。
`;


try {
  await runAgentWithTools(query);
} catch (error) {
  console.error(`\n❌ 错误: ${error.message}\n`);
}
