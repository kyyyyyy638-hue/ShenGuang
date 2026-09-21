import { tool } from '@langchain/core/tools';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { z } from 'zod';

// 平台信息 —— 注入到工具 description 里给模型看。
// 不声明平台的话，模型会按 Unix 习惯编命令（ls / mkdir -p），在 Windows 上全部失效
const IS_WIN = process.platform === 'win32';
const PLATFORM_HINT = IS_WIN
  ? '当前是 Windows 环境（shell 是 cmd.exe）：用 dir 不要用 ls，用 type 不要用 cat，用 findstr 不要用 grep，创建目录用 mkdir 不要用 mkdir -p，删除目录用 rmdir /s /q。'
  : '当前是 Unix 环境（shell 是 /bin/sh）。';

// 1. 读取文件工具
const readFileTool = tool(
  async ({ filePath }) => {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      console.log(`  [工具调用] read_file("${filePath}") - 成功读取 ${content.length} 个字符`);
      return `文件内容:\n${content}`;
    } catch (error) {
      console.log(`  [工具调用] read_file("${filePath}") - 错误: ${error.message}`);
      return `读取文件失败: ${error.message}`;
    }
  },
  {
    name: 'read_file',
    description:
      '用此工具来读取文件内容。当用户要求读取文件、查看代码、分析文件内容时，调用此工具。输入文件路径（可以是相对路径或绝对路径）。',
    schema: z.object({
      filePath: z.string().describe('要读取的文件路径'),
    }),
  }
);

// 2. 写入文件工具
const writeFileTool = tool(
  async ({ filePath, content }) => {
    try {
      const dir = path.dirname(filePath);
      await fs.mkdir(dir, { recursive: true });
      await fs.writeFile(filePath, content, 'utf-8');
      console.log(`  [工具调用] write_file("${filePath}") - 成功写入 ${content.length} 个字符`);
      return `文件写入成功: ${filePath}`;
    } catch (error) {
      console.log(`  [工具调用] write_file("${filePath}") - 错误: ${error.message}`);
      return `写入文件失败: ${error.message}`;
    }
  },
  {
    name: 'write_file',
    description:
      '用此工具把内容写入文件，父目录不存在时自动创建。当用户要求创建文件、写代码、修改文件内容时调用。注意：会覆盖同名文件。',
    schema: z.object({
      filePath: z.string().describe('要写入的文件路径'),
      content: z.string().describe('要写入的完整文件内容'),
    }),
  }
);

// ── 命令执行内部函数 ─────────────────────────────────────────
// agent 工具必须把输出抓回来（return 给模型），不能用 stdio: 'inherit'
// —— inherit 会把输出直接漏到终端，程序一个字都拿不到。
// 两种模式的 stdio 用法完全相反，捕获代码不能共用：
//   前台（默认）：走管道 → 能抓到 stdout/stderr，但「进程退出」≠「管道关闭」
//   后台（background）：stdio 全 ignore → 不建管道（没有句柄可泄漏），立即返回，拿不到输出
function runCommand(command, cwd, { timeout = 120_000, background = false } = {}) {
  // Windows 上先 chcp 65001 把代码页切到 UTF-8，再执行真正的命令：
  // - cmd 内部命令（dir/type/echo）按「控制台代码页」编码输出 → 切成 UTF-8
  // - node/pnpm/git 等外部程序往管道写的本来就是 UTF-8
  //   （实测：不切的话，type 读 UTF-8 文件、node 的中文输出都会乱码，
  //    因为那些字节会被按 GBK 解释 —— 「这是写入测试」会变成「杩欐槸鍐欏叆娴嬭瘯」）
  // 整条管道统一 UTF-8 之后，解码层就简单了
  const finalCommand =
    process.platform === 'win32' ? `chcp 65001 > nul && ${command}` : command;

  // ── 后台模式：启动常驻服务（dev server / watch）用这个 ──
  // 为什么必须 detached + stdio: 'ignore'：
  //   普通 spawn 建的 stdout 管道，写端会被子进程继承。后台命令的子孙进程攥着这个
  //   写端不放，父进程侧就永远读不到 EOF，'close' 事件不触发 —— agent 卡在 await 上
  //   （实测：`start /b` 起的子进程活 7s，close 就晚 7s；常驻服务则永远不来）。
  //   stdio: 'ignore' 让 Node 根本不建管道，从根上消除这个泄漏。
  // 代价：拿不到任何输出，所以 child.stdout / child.stderr 都是 null，绝不能访问。
  if (background) {
    const bg = spawn(finalCommand, {
      cwd,
      shell: true,
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
    });
    // error 不监听会直接崩掉整个进程（EventEmitter 无监听器时抛错）
    bg.on('error', (error) => {
      console.log(`  [后台进程] 启动失败: ${error.message}`);
    });
    // unref：父进程不等它，事件循环也不会被它挂住
    bg.unref();
    return Promise.resolve({
      ok: true,
      code: 0,
      text: `已在后台启动（pid=${bg.pid}），输出不会被捕获。\n请用其它手段确认是否就绪（例如另跑一条命令做端口探测，或直接用 HTTP 请求访问），不要试图等待它退出。`,
    });
  }

  // ── 前台模式：要拿到输出，就必须走管道 ──
  return new Promise((resolve) => {
    const child = spawn(finalCommand, { cwd, shell: true, windowsHide: true });

    const decOut = new TextDecoder('utf-8');
    const decErr = new TextDecoder('utf-8'); // stdout / stderr 各用一个实例（stream 模式下有状态）

    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let settled = false;
    let spawnError = null;
    let timer = null;
    let drainTimer = null;

    // 管道是否已正常 EOF —— 没 EOF 说明有子孙进程还攥着写端，输出可能不全
    const pipeClosed = () => child.stdout.readableEnded && child.stderr.readableEnded;

    const buildResult = (code) => {
      if (spawnError) {
        return { ok: false, code: -1, text: `命令启动失败: ${spawnError}` };
      }
      if (timedOut) {
        return {
          ok: false,
          code,
          text: `命令超时（${timeout}ms）已强制终止\n\nstdout:\n${stdout}\nstderr:\n${stderr}`,
        };
      }
      const warn = pipeClosed()
        ? ''
        : '\n\n注意：命令已退出，但输出管道未正常关闭（多半是有后台子进程仍持有它），以上输出可能不完整。';
      // 三段式：退出码 + stdout + stderr，模型靠它判断成功与否、读错误、重试
      return {
        ok: code === 0,
        code,
        text: `exit code: ${code}\n\nstdout:\n${stdout || '(空)'}\n\nstderr:\n${stderr || '(空)'}${warn}`,
      };
    };

    // 幂等结算：close / exit 兜底 / 超时兜底三条路谁先到谁生效，只 resolve 一次
    const finish = (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearTimeout(drainTimer);
      resolve(buildResult(code));
    };

    child.stdout.on('data', (c) => {
      stdout += decOut.decode(c, { stream: true }); // stream:true 处理被 chunk 边界切断的多字节字符
    });
    child.stderr.on('data', (c) => {
      stderr += decErr.decode(c, { stream: true });
    });

    child.on('error', (error) => {
      // 进程根本没启动起来（可执行文件不存在、没权限）
      // 这里必须结算，否则 Promise 永远挂起，agent 直接卡死
      spawnError = error.message;
      finish(-1);
    });

    // ★ 结算不能只挂 close：
    //   'close' 的触发条件是「进程退出 + stdout/stderr 管道 EOF」。正常命令两者几乎同时，
    //   但只要有一个子孙进程继承了管道写端，close 就会无限延后（这是上一版卡死的根因）。
    //   这里 exit 后留 500ms 排空缓冲：正常命令由 close 先到、拿完整输出；
    //   异常情况由定时器兜底结算，agent 不会卡死。
    child.on('exit', (code) => {
      drainTimer = setTimeout(() => finish(code), 500);
    });
    child.on('close', (code) => finish(code));

    // 有些命令永不退出（ping、起服务、npm run dev），必须兜底
    timer = setTimeout(() => {
      timedOut = true;
      if (process.platform === 'win32') {
        // Windows 上 child.kill() 只杀 shell 本身，杀不掉子进程树
        spawn(`taskkill /pid ${child.pid} /T /F`, { shell: true, windowsHide: true });
      } else {
        child.kill();
      }
      // 再兜一层：进程树杀不掉、管道也不关的话，也不能让 agent 卡住
      setTimeout(() => finish(null), 500);
    }, timeout);
  });
}

// 3. 执行命令工具
const executeCommandTool = tool(
  async ({ command, workingDirectory, background }) => {
    const cwd = workingDirectory || process.cwd();
    console.log(`  [工具调用] execute_command("${command}")${background ? ' [后台]' : ''}${workingDirectory ? ` - 工作目录: ${workingDirectory}` : ''}`);

    const result = await runCommand(command, cwd, { background });
    console.log(`  [工具调用] execute_command("${command}") - ${result.ok ? '执行成功' : `失败，退出码: ${result.code}`}`);

    const cwdInfo = workingDirectory
      ? `\n\n重要提示：本次命令的工作目录是 "${workingDirectory}"。继续在这个目录执行命令时，请使用 workingDirectory: "${workingDirectory}" 参数，不要在 command 里写 cd。`
      : '';

    return result.text + cwdInfo;
  },
  {
    name: 'execute_command',
    description: `执行系统命令并返回它的完整输出（stdout / stderr / 退出码）。${PLATFORM_HINT} 用 workingDirectory 参数指定工作目录，不要在 command 里写 cd。
启动常驻服务（dev server、watch 模式）必须传 background: true —— 前台执行要等命令退出才返回，而常驻命令永不退出，会一直阻塞到 120 秒超时。background: true 会立即返回且不捕获输出，之后请用端口检查或 HTTP 请求确认服务是否就绪。不要用 "start /b xxx"、"xxx &" 之类的方式自己后台化。`,
    schema: z.object({
      command: z.string().describe('要执行的完整命令字符串'),
      workingDirectory: z.string().optional().describe('命令的工作目录（推荐指定）'),
      background: z
        .boolean()
        .optional()
        .describe(
          '是否后台启动常驻服务，默认 false。仅用于 dev server / watch 这类永不退出的命令；开启后立即返回且拿不到输出'
        ),
    }),
  }
);

// 4. 列出目录内容工具
const listDirectoryTool = tool(
  async ({ directoryPath }) => {
    try {
      // withFileTypes 能区分文件和目录，模型拿到这个信息才好决定下一步
      const files = await fs.readdir(directoryPath, { withFileTypes: true });
      const lines = files.map((f) => `${f.isDirectory() ? '[目录]' : '[文件]'} ${f.name}`);
      console.log(`  [工具调用] list_directory("${directoryPath}") - 找到 ${files.length} 个项目`);
      return `目录内容:\n${lines.join('\n')}`;
    } catch (error) {
      console.log(`  [工具调用] list_directory("${directoryPath}") - 错误: ${error.message}`);
      return `列出目录失败: ${error.message}`;
    }
  },
  {
    name: 'list_directory',
    description:
      '列出指定目录下的所有文件和文件夹。当用户要求查看目录结构、确认文件是否存在时调用。',
    schema: z.object({
      directoryPath: z.string().describe('目录路径（可以是相对路径或绝对路径）'),
    }),
  }
);

export { readFileTool, writeFileTool, executeCommandTool, listDirectoryTool };
