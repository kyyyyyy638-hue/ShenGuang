import { fileURLToPath } from 'node:url';
import 'dotenv/config';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { ChatOpenAI } from '@langchain/openai';
import chalk from 'chalk';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';

// 模型配置对齐 min-cursor.mjs：GLM-5.3 必须显式开 thinking
const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL },
  modelKwargs: {
    thinking: { type: 'enabled' },
    reasoning_effort: process.env.REASONING_EFFORT || 'low',
  },
});

const mcpClient = new MultiServerMCPClient({
    mcpServers: {
        'my-mcp-server': {
            command: "node",
            args: [
                fileURLToPath(new URL('./my-mcp-server.mjs', import.meta.url))
            ]
        }
    }
});




const tools = await mcpClient.getTools();
const modelWithTools = model.bindTools(tools);

function stringify(value) {
    if (typeof value === 'string') return value;
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

// MCP 的结果可能是 { content: [{ type: 'text', text }] } 也可能是裸字符串/数组，统一拍平
function stringifyToolResult(raw) {
    if (typeof raw === 'string') return raw;
    const maybeBlocks = Array.isArray(raw) ? raw : raw?.content;
    if (Array.isArray(maybeBlocks)) {
        return maybeBlocks.map((b) => (b?.type === 'text' ? b.text : stringify(b))).join('\n');
    }
    return stringify(raw);
}

// 每轮请求都会重新带上的 tools 定义（模型能"感知"到的全部东西就在这里）
function dumpTools(toolList) {
    console.log(chalk.magenta('──────── tools（本轮随请求发出）────────'));
    for (const t of toolList) {
        console.log(chalk.magenta(`name: ${t.name}`));
        console.log(chalk.magenta(`description: ${t.description}`));
        console.log(chalk.magenta(`parameters(JSON Schema):\n${stringify(t.schema)}`));
    }
}

// 本轮真正发给模型的 messages
function dumpMessages(messageList) {
    console.log(chalk.cyan('──────── messages（本轮发给模型）────────'));
    messageList.forEach((m, idx) => {
        const role = typeof m._getType === 'function' ? m._getType() : m.role;
        console.log(chalk.cyan(`[${idx}] ${role}` + (m.tool_call_id ? ` (tool_call_id: ${m.tool_call_id})` : '')));
        if (m.tool_calls?.length) {
            console.log(chalk.yellow(`tool_calls:\n${stringify(m.tool_calls)}`));
        }
        const content = typeof m.content === 'string' ? m.content : stringifyToolResult(m.content);
        if (content) console.log(chalk.cyan(`content:\n${content}`));
    });
}

async function runAgentWithTools(query, maxIterations = 30) {
    const messages = [
        new HumanMessage(query)
    ];

    for (let i = 0; i < maxIterations; i++) {
        console.log(chalk.inverse(`\n=================== 第 ${i + 1} 轮 ===================`));
        dumpTools(tools);
        dumpMessages(messages);

        console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));
        const response = await modelWithTools.invoke(messages);
        messages.push(response);

        console.log(chalk.bgBlue('\n──────── 模型本轮回复 ────────'));
        console.log(chalk.bgBlue(`content: ${stringifyToolResult(response.content) || '(空)'}`));
        if (response.tool_calls?.length) {
            console.log(chalk.bgBlue(`tool_calls:\n${stringify(response.tool_calls)}`));
        }

        // 检查是否有工具调用
        if (!response.tool_calls || response.tool_calls.length === 0) {
            console.log(`\n✨ AI 最终回复:\n${response.content}\n`);
            return response.content;
        }

        console.log(chalk.bgBlue(`🔍 检测到 ${response.tool_calls.length} 个工具调用`));
        console.log(chalk.bgBlue(`🔍 工具调用: ${response.tool_calls.map(t => t.name).join(', ')}`));
        // 执行工具调用
        for (const toolCall of response.tool_calls) {
            const foundTool = tools.find(t => t.name === toolCall.name);
            let toolResult;
            if (!foundTool) {
                // 找不到也要回一条，否则下一轮会因为 tool_calls 没有响应而报协议错误
                toolResult = `错误: 找不到名为 "${toolCall.name}" 的工具。可用工具: ${tools.map((t) => t.name).join(', ')}`;
            } else {
                try {
                    toolResult = stringifyToolResult(await foundTool.invoke(toolCall.args));
                } catch (error) {
                    toolResult = `工具执行出错: ${error.message}`;
                }
            }

            console.log(chalk.bgYellow('\n──────── 工具执行结果（即将作为 ToolMessage 回写）────────'));
            console.log(chalk.yellow(`tool_call_id: ${toolCall.id}`));
            console.log(chalk.yellow(`${toolResult}`));

            messages.push(new ToolMessage({
                content: toolResult,
                tool_call_id: toolCall.id,
            }));
        }
    }

    return messages[messages.length - 1].content;
}

await runAgentWithTools("查一下用户 002 的信息");


/*const res = await mcpClient.listResources();

for (const [serverName, resources] of Object.entries(res)) {
    for (const resource of resources) {
        const content = await mcpClient.readResource(serverName, resource.uri);
        console.log(content);
    }
}*/

// 关掉 client —— 连带杀掉它 spawn 的 server 子进程，否则 node 不退出
await mcpClient.close()