// ============================================================
// 【这个文件干什么】
// 往 Milvus（一个专门存"向量"的数据库）里插入 5 条日记数据。
//
// 【运行方式】必须在 milvus-test 目录下执行（因为 .env 是按当前目录找的）：
//     cd E:\代码项目\ShenGuang\milvus\milvus-test
//     node src/insert.mjs
//
// 【前置条件】
//   1. Milvus 已经跑起来（Docker 容器，监听 19530 端口）
//   2. 同目录有 .env 文件，里面配好了 Embedding 服务的 key 和地址
//
// 【先记住三个词，后面全靠它们】
//   · 向量（vector）：一段文字丢给模型，模型吐出来的一串数字，比如 1024 个数。
//     意思相近的文字，算出来的数字串也相近 —— 计算机靠这个"懂"语义。
//   · 集合（collection）：相当于 MySQL 里的"表"。这里建了一张叫 ai_diary 的表。
//   · 索引（index）：让查询变快的数据结构，相当于书的目录。不建也能存，但查得慢。
// ============================================================

// dotenv/config：读当前目录下的 .env 文件，把里面的内容塞进 process.env。
// 不写这一行，下面 process.env.OPENAI_API_KEY 拿到的就是 undefined。
import "dotenv/config";

// 从隔壁 models.mjs 借来"向量化工具"。
// 为什么不在这里自己 new 一个？因为换模型时只改 models.mjs 一处就行，
// 否则每个脚本都要改一遍，很容易漏。
import { embeddings } from "./models.mjs";

// 从 Milvus 官方 SDK 里拿 4 样东西：
//   MilvusClient —— 客户端，所有和数据库的交互都通过它
//   DataType     —— 字段类型枚举，相当于 SQL 的 VARCHAR / INT
//   MetricType   —— 判断"两个向量有多像"的算法，这里用余弦相似度
//   IndexType    —— 索引类型，这里用最朴素的 FLAT
import {
  MilvusClient,
  DataType,
  MetricType,
  IndexType,
} from "@zilliz/milvus2-sdk-node";

// 集合名，就是"表名"。
// 优先读 .env 里的 MILVUS_COLLECTION，没配就用默认的 ai_diary。
// ?? 的意思是"左边是 null/undefined 时才用右边"，和 || 的区别是空字符串也算有效值。
const COLLECTION_NAME = process.env.MILVUS_COLLECTION ?? "ai_diary";

// 向量维度 = 模型吐出来的数字串有多少个数。
// 【极易踩的坑】这个数字必须和 .env 里 EMBEDDING_MODEL 的实际输出一致，否则插不进去：
//     BAAI/bge-m3         -> 1024 维
//     text-embedding-v3   -> 1024 维（默认，可调）
// 如果换了别的模型（比如 768 维的），这里必须跟着改。
const VECTOR_DIM = 1024;

// 创建客户端。注意：new 的时候只是准备好配置，还没真正连上。
// 真正建立连接是下面 main() 里的 await client.connectPromise。
const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS ?? "localhost:19530", // 数据库地址：本机 19530 端口
});

// 准备要插入的数据。每条是一个对象，字段名必须和下面建表时的字段名一一对应。
// 现在这里故意没写 vector 字段 —— 因为向量还没算出来，后面会补上。
const diaryEntries = [
  {
    id: "diary_001", // 主键，每条不能重复
    content: "今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。",
    date: "2026-01-10",
    mood: "happy",
    tags: ["生活", "散步"], // 数组字段，最多放 10 个
  },
  {
    id: "diary_002",
    content: "今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。",
    date: "2026-01-11",
    mood: "excited",
    tags: ["工作", "成就"],
  },
  {
    id: "diary_003",
    content: "周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。",
    date: "2026-01-12",
    mood: "relaxed",
    tags: ["户外", "朋友"],
  },
  {
    id: "diary_004",
    content: "今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。",
    date: "2026-01-12",
    mood: "curious",
    tags: ["学习", "技术"],
  },
  {
    id: "diary_005",
    content: "晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。",
    date: "2026-01-13",
    mood: "proud",
    tags: ["美食", "家庭"],
  },
];

// async 表示这是个"异步函数"：里面凡是耗时操作（发网络请求、读写数据库）
// 前面都要加 await，让程序停下来等结果，而不是硬冲过去拿到一个空值。
async function main() {
  // ---------- 第 1 步：连数据库 ----------
  // connectPromise 是 SDK 内部的连接承诺，await 它会等连接真正建立好。
  await client.connectPromise;
  console.log(`已连接 Milvus: ${process.env.MILVUS_ADDRESS ?? "localhost:19530"}`);

  // ---------- 第 2 步：如果旧集合还在，先删掉 ----------
  // 为什么要删？因为 Milvus 不允许同名集合重复创建，第二次运行会直接报错。
  // 先删再建，脚本就能反复跑，每次都是干净的 5 条数据。
  // hasCollection 返回 { value: true/false }，这里用解构把 value 取出来改名成 exists。
  const { value: exists } = await client.hasCollection({
    collection_name: COLLECTION_NAME,
  });
  if (exists) {
    await client.dropCollection({ collection_name: COLLECTION_NAME });
    console.log(`集合 ${COLLECTION_NAME} 已存在，已删除并重建`);
  }

  // ---------- 第 3 步：建"表"（集合） ----------
  // fields 数组就是这张表的列定义，一列一个对象：
  //   name        列名（和插入数据的字段名对应）
  //   data_type   数据类型
  //   max_length  VarChar 最长多少个字符，超了会插入失败
  //   is_primary_key  标记主键（相当于身份证号，不能重复）
  await client.createCollection({
    collection_name: COLLECTION_NAME,
    fields: [
      // 主键：50 字符以内的字符串
      { name: "id", data_type: DataType.VarChar, max_length: 50, is_primary_key: true },

      // 【最重要的列】存向量的特殊列，必须能容纳 VECTOR_DIM 个数
      { name: "vector", data_type: DataType.FloatVector, dim: VECTOR_DIM },

      // 正文，给到 5000 字符，避免长日记被截断
      { name: "content", data_type: DataType.VarChar, max_length: 5000 },
      { name: "date", data_type: DataType.VarChar, max_length: 50 },
      { name: "mood", data_type: DataType.VarChar, max_length: 50 },

      // 数组列：element_type 说明每个元素是字符串，max_capacity 是最多几个元素
      {
        name: "tags",
        data_type: DataType.Array,
        element_type: DataType.VarChar,
        max_capacity: 10,
        max_length: 50,
      },
    ],
  });
  console.log(`集合 ${COLLECTION_NAME} 已创建`);

  // ---------- 第 4 步：给向量列建索引 ----------
  // 为什么只给 vector 建？因为向量检索是逐个算相似度的，数据多了特别慢，
  // 必须靠索引加速。其他列（id/mood 之类）是普通字段，不需要。
  //
  // index_type = FLAT：暴力精确比对，不做任何近似。适合小数据量（几千条以内）。
  //   教程里用的 IVF_FLAT + nlist=1024 是为大数据集设计的，
  //   当你只有 5 条数据、nlist 却是 1024 时，会触发 "nlist should be less than row count" 警告。
  // metric_type = COSINE：用余弦夹角判断两个向量像不像，越接近 1 越相似。
  //   这是文本场景最常用的算法。
  await client.createIndex({
    collection_name: COLLECTION_NAME,
    field_name: "vector",
    index_type: IndexType.FLAT,
    metric_type: MetricType.COSINE,
  });
  console.log("向量索引已创建 (FLAT / COSINE)");

  // ---------- 第 5 步：把集合加载进内存 ----------
  // Milvus 的查询是在内存里做的，集合建好后默认在磁盘上，必须 load 才能被搜索。
  // 【新手最容易忘的一步】忘了 load 就去查，会报 collection not loaded。
  await client.loadCollection({ collection_name: COLLECTION_NAME });
  console.log("集合已加载到内存");

  // ---------- 第 6 步：把文字变成向量 ----------
  // embedDocuments 会一次把这 5 段文字发给模型，返回 5 个向量（每个是 1024 个数字的数组）。
  // map 的作用是从每条日记里只挑出 content 字段，拼成一个纯文本数组。
  // 注意用的是 embedDocuments（批量、给文档用），不是 embedQuery（单条、给提问用）。
  console.log(`\n开始向量化 ${diaryEntries.length} 条数据...`);
  const vectors = await embeddings.embedDocuments(
    diaryEntries.map((entry) => entry.content),
  );

  // 先验证模型实际给了多少维。万一 .env 里的模型换了、维度变成 768，
  // 而集合是按 1024 建的，插入会失败且报错很晦涩。这里提前拦一下，报错才看得懂。
  const actualDim = vectors[0]?.length;
  if (actualDim !== VECTOR_DIM) {
    throw new Error(
      `向量维度不匹配：模型返回 ${actualDim} 维，集合定义 ${VECTOR_DIM} 维。` +
        `请改 VECTOR_DIM 后重跑（集合已按新维度重建）。`,
    );
  }
  console.log(`向量化完成，维度 ${actualDim}`);

  // ---------- 第 7 步：真正写入数据 ----------
  // 把原来的日记对象和算好的向量拼在一起（...entry 是"展开"原对象的所有字段），
  // vector 字段就是第 6 步得到的那串数字。返回的 insert_cnt 是成功条数。
  const { insert_cnt } = await client.insert({
    collection_name: COLLECTION_NAME,
    data: diaryEntries.map((entry, i) => ({ ...entry, vector: vectors[i] })),
  });

  // ---------- 第 8 步：刷盘 ----------
  // Milvus 写入是"先攒着、后落盘"的，不 flush 的话刚插的数据可能立刻查不到。
  // 【坑】SDK 3.x 这里必须传 collection_names（复数、数组），
  //      传 collection_name 会报 "The `collection_name` property is missing."，
  //      这个报错极具误导性 —— 其实是参数名写错了，不是集合名没给。
  await client.flush({ collection_names: [COLLECTION_NAME] });

  console.log(`\n插入完成：${insert_cnt} 条 -> ${COLLECTION_NAME}`);
}

// 调用上面的函数，开始干活。
// .catch 负责兜住任何一步抛出的错误：打印人话版报错，并以失败状态退出。
//   process.exit(1) 里的 1 表示"程序出错了"，0 表示成功。
// .finally 不管成功还是失败都会执行：关掉数据库连接，避免进程挂在那里不退出。
main()
  .catch((error) => {
    console.error("\n失败:", error.message);
    process.exit(1);
  })
  .finally(async () => {
    await client.closeConnection();
  });
