// 工具集自测 —— 不经过模型、不需要 API Key、不花 token
// 验证 4 个工具在当前平台上是否真的能工作（md2 的 self-test-tool.mjs 只测了 read_file 一个）
// 运行：node ./src/self-test-tools.mjs
import { readFileTool, writeFileTool, executeCommandTool, listDirectoryTool } from './all-tools.mjs';

console.log('═══ 1. list_directory ═══');
console.log(await listDirectoryTool.invoke({ directoryPath: './src' }));

console.log('\n═══ 2. write_file ═══');
console.log(
  await writeFileTool.invoke({
    filePath: './.smoke-test.txt',
    content: '这是写入测试\n第二行 中文内容',
  })
);

console.log('\n═══ 3. read_file（读回来验证内容一致）═══');
console.log(await readFileTool.invoke({ filePath: './.smoke-test.txt' }));

console.log('\n═══ 4. execute_command（用命令读同一个文件，顺便验证输出捕获与编码）═══');
const cmd = process.platform === 'win32' ? 'type .smoke-test.txt' : 'cat .smoke-test.txt';
console.log(await executeCommandTool.invoke({ command: cmd }));

// 清理测试文件
const fs = await import('node:fs/promises');
await fs.unlink('./.smoke-test.txt');
console.log('\n（已清理 .smoke-test.txt）');
