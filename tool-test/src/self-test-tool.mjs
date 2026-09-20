// 工具自测 —— 不经过模型、不需要 API Key、不花任何 token
// 用途：先把「tool 本身写对了没有」验证掉，再去接模型
// 运行：node ./src/self-test-tool.mjs
import { readFileTool } from './tools/read-file.mjs';

console.log('工具名:', readFileTool.name);
console.log('工具描述:', readFileTool.description);
console.log('参数 schema:', JSON.stringify(readFileTool.schema, null, 2));
console.log('\n--- 真实调用一次，读它自己 ---');

const result = await readFileTool.invoke({ filePath: './package.json' });
console.log(result.slice(0, 300));
