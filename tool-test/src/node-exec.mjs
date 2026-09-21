import { spawn } from 'node:child_process';

// ─────────────────────────────────────────────────────────────
// 改动 1：平台差异
//
// 教程作者用 Mac/Linux，那边有 `ls`；Windows 的 cmd.exe 里没有 `ls`，
// 对应命令是 `dir`。因为开了 shell: true，实际执行的 shell 是：
//     Windows → cmd.exe        Mac/Linux → /bin/sh
// 所以照抄原版在 Windows 上会报「'ls' 不是内部或外部命令」。
// 注意：那句报错是 cmd.exe 输出的，不是 Node —— 进程本身启动成功了。
// ─────────────────────────────────────────────────────────────
const command = process.platform === 'win32' ? 'dir' : 'ls -la';

const cwd = process.cwd();

// ─────────────────────────────────────────────────────────────
// 改动 2：去掉手动的 command.split(' ')
//
// 教程里有 `const [cmd, ...args] = command.split(' ')`。但既然已经
// shell: true，解析命令就是 shell 的活儿 —— 手动拆开再让 Node 拼回去
// 属于重复劳动，遇到引号或带空格的路径还会拆错，并触发 DEP0190 警告
// （警告含义：开启 shell 时传给子进程的参数不会被转义，只有拼接）。
//
// 直接把整条命令字符串交给 spawn 即可：第二参数传对象时 args 可省略。
// ─────────────────────────────────────────────────────────────
const child = spawn(command, {
  cwd,
  stdio: 'inherit', // 子进程的 stdout/stderr 直接接到父进程终端
  shell: true, // 套一层 shell 执行（Windows=cmd.exe，Unix=/bin/sh）
});

let errorMsg = '';

// error 事件 = 进程根本没启动起来
// （类比 Java：ProcessBuilder.start() 抛 IOException）
// 注意：「命令不存在」不走这里 —— 那种情况启动的是 shell 本身，而 shell 启动成功了
child.on('error', (error) => {
  errorMsg = error.message;
});

// close 事件 = 进程正常结束，带一个退出码
// （类比 Java：Process.waitFor() 的返回值）
// 「命令本身跑失败了」属于这里 —— 进程活得好好的，只是退出码非零

child.on('close', (code) => {
if (code === 0) {
    process.exit(0);
  } else {
    if (errorMsg) {
      console.error(`错误: ${errorMsg}`);
    }
    process.exit(code || 1);
  }
});