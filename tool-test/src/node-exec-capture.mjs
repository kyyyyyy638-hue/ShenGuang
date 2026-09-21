// 版本二：把命令输出「抓回来」，而不是直接打到终端。
//
// 为什么需要这一版？
//   md3 的 execute_command 工具要把命令结果 return 出去，塞进 ToolMessage
//   发给模型。而 stdio: 'inherit' 意味着输出流向父进程的终端，程序一个字
//   都拿不到 —— 所以 agent 工具必须用监听 data 事件的方式捕获。
//
// 运行：node ./src/node-exec-capture.mjs
import { spawn } from 'node:child_process';

function runCommand(command, { cwd = process.cwd(), timeout = 30_000 } = {}) {
  return new Promise((resolve) => {
    // 注意这里没有 stdio: 'inherit' —— 否则收不到 data 事件
    const child = spawn(command, { cwd, shell: true });

    // ⚠️ 编码坑（Windows 专有，但很致命）
    //    cmd.exe 输出的字节流是 GBK（代码页 936），而 Node 的 chunk.toString()
    //    默认按 UTF-8 解码 → 中文全变乱码（实测：`你好` 会变成 `���`）。
    //    更要命的是这段内容要发给模型，乱码模型完全读不懂，只能瞎猜。
    //
    //    按平台选解码器。stdout / stderr 必须各用一个实例 ——
    //    TextDecoder 在 stream 模式下是有状态的，共用一个会把两条流串起来。
    const enc = process.platform === 'win32' ? 'gbk' : 'utf-8';
    const decOut = new TextDecoder(enc);
    const decErr = new TextDecoder(enc);

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    // { stream: true } 让解码器缓存被 chunk 边界切断的多字节字符，
    // 否则一个汉字正好被切成两半时会解码失败
    child.stdout.on('data', (chunk) => {
      stdout += decOut.decode(chunk, { stream: true });
    });
    child.stderr.on('data', (chunk) => {
      stderr += decErr.decode(chunk, { stream: true });
    });

    // 有些命令永远不会自己退出（ping、起服务、npm run dev），必须兜个超时
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, timeout);

    // error = 进程没能启动（可执行文件不存在、没权限）
    child.on('error', (error) => {
      clearTimeout(timer);
      resolve(`命令启动失败: ${error.message}`);
    });

    // close = 进程结束，带上退出码
    child.on('close', (code) => {
      clearTimeout(timer);

      if (timedOut) {
        resolve(`命令超时（${timeout}ms）已终止\n\nstdout:\n${stdout}\nstderr:\n${stderr}`);
        return;
      }

      // ★ 这个字符串就是最终要 return 给模型的内容。
      //   把 stdout / stderr / exit code 都给出去，模型才能自己判断成功与否、
      //   并在失败时读错误信息、换命令重试。
      resolve(`exit code: ${code}\n\nstdout:\n${stdout}\nstderr:\n${stderr}`);
    });
  });
}

// 跨平台：Windows 用 dir，Unix 用 ls
const command = process.platform === 'win32' ? 'dir' : 'ls -la';

const result = await runCommand(command);
console.log('───── 捕获到的内容（原本要发给模型的就是这段）─────');
console.log(result);
