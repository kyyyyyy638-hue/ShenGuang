// ============================================================
// 【这个文件干什么】
// 把一整本电子书切碎、向量化，灌进 Milvus，供后面做语义检索。
//
// 为什么需要这一步：
//   问"段誉会什么武功"，用 MySQL 的 LIKE 根本没法查 —— 你连关键词都不知道该定什么。
//   搜"段誉"会命中所有出现这两个字的地方（可能几百处，大部分是废话）；
//   搜"武功"更糟，全书都在讲武功。
//   而向量检索是拿"问题的语义"去比对"每段文字的语义"，
//   不需要你事先想好关键词。
//
// 【运行方式】在 milvus-test 目录下：
//     cd E:\代码项目\ShenGuang\milvus\milvus-test
//     node src/ebook-writer.mjs
//
// 【电子书放哪】
//   默认读 data/ebook.txt（UTF-8 编码的纯文本）。
//   文件不存在时会自动用内置演示文本跑通流程，方便先看效果。
//   想指定别的文件，用环境变量：
//     $env:EBOOK_PATH="E:\book\天龙八部.txt"; node src/ebook-writer.mjs
//
// 【为什么必须切块（chunk）】
//   一本小说几十万字，不可能整本塞进一次请求（模型有输入上限），
//   也不可能把整本存成一条向量 —— 那样"整本书的语义"会模糊成一团，
//   检索出来只能定位到"某本书"，无法定位到"某一段"。
//   切成 500 字左右的小块，每块单独算向量，检索才能精确定位到具体段落。
// ============================================================

import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  MilvusClient,
  DataType,
  IndexType,
  MetricType,
} from "@zilliz/milvus2-sdk-node";

// 文本切分器：把一个长字符串按规则切成多个小块。
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { embeddings } from "./models.mjs";

// ---------- 配置 ----------

const COLLECTION_NAME = process.env.EBOOK_COLLECTION ?? "ebook_chunks";

// 必须和 .env 里 EMBEDDING_MODEL 的实际输出维度一致（bge-m3 / text-embedding-v3 都是 1024）
const VECTOR_DIM = 1024;

// 书名会作为一列存进去，方便以后多本书共用一个集合
const BOOK_NAME = process.env.BOOK_NAME ?? "天龙八部";

// 每块多少字、相邻块重叠多少字。
//   chunk_size 太小 → 一句话被切断，语义不完整
//   chunk_size 太大 → 一块里混着好几件事，检索精度下降
//   500 字对中文小说是个稳妥值；overlap 50 是为了避免恰好把关键句切在边界上
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 50;

// 每批发给 embedding 接口多少条。太大容易触发限流或超时，
// 太小则请求次数过多（每次都有网络往返开销）。
const EMBED_BATCH = 10;

// 每批写入 Milvus 多少行
const INSERT_BATCH = 50;

// 脚本在 src/ 下，所以 data 目录要往上一级找
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_EBOOK_PATH = path.resolve(__dirname, "../data/ebook.txt");
const EBOOK_PATH = process.env.EBOOK_PATH
  ? path.resolve(process.env.EBOOK_PATH)
  : DEFAULT_EBOOK_PATH;

const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS ?? "localhost:19530",
});

// ---------- 内置演示文本 ----------
// 仅用于"没有电子书文件时也能跑通流程"，内容是自己组织的概述，
// 不是原著原文。验证完流程后请换成真实电子书。
const DEMO_TEXT = `段誉是大理国镇南王段正淳之子，从小不喜习武，只爱读书，被父亲逼着练剑时常偷懒。
他因不愿学武离家出走，误入无量山，在琅嬛玉洞中见到了北冥神功与凌波微步两门绝学的图谱。
北冥神功能吸取他人内力化为己用，段誉起初不肯修炼，觉得这是损人利己的功夫，只在危急关头无意中使出。
凌波微步是一门轻功，按易经六十四卦的方位行走，脚下位置全凭卦象推算，身形飘忽不定，敌人极难近身。
后来段誉在天龙寺被鸠摩智擒住，辗转途中又习得六脉神剑。
六脉神剑以手指发出无形剑气，分为少商、商阳、中冲、关冲、少泽、少冲六路剑法，以内力催动，锋利无比。
段誉的六脉神剑时灵时不灵，因为他不愿与人争斗，内力运转不畅。
在少室山一战中，段誉终于以内力催动六脉神剑，击败了慕容复。
他后来继承大理皇位，成为大理国君主，仍不改仁厚本性。

乔峰是丐帮帮主，武功以刚猛著称，最出名的是降龙十八掌，掌力雄浑，一掌拍出有开碑裂石之势。
他还会打狗棒法，这是丐帮帮主的信物绝学，招式以巧取胜，专攻敌人下盘。
乔峰身世坎坷，后来得知自己是契丹人，被迫离开丐帮，改名萧峰。
他在聚贤庄一役中以一敌众，用降龙十八掌连伤数十位中原武林高手。

虚竹本是少林寺的一个小和尚，相貌平凡，性格木讷，一心只想当个好和尚。
他误打误撞破了珍珑棋局，被逍遥派掌门无崖子收为弟子，继承了七十余年内力。
之后虚竹成为灵鹫宫主人，学会了天山六阳掌、天山折梅手和生死符。
生死符是一种用内力凝成的薄冰，打入人体后发作时奇痒剧痛，是灵鹫宫控制部属的手段。

王语嫣是姑苏曼陀山庄的大小姐，她不会武功，却熟知天下各门各派的武功秘籍，
能一眼看出对方招式的来历和破绽。段誉在无量山见到她之后一见倾心，一路追随，
她心里却只有表哥慕容复。

慕容复是姑苏慕容氏的传人，人称北乔峰南慕容，家传绝学是斗转星移。
斗转星移的要点是以彼之道还施彼身，把对方的力道引向他处，让对手打在自己身上。
他一心想着复兴大燕，为此不择手段，最终众叛亲离。

丁春秋是星宿派掌门，擅长化功大法，能化去对手内力，为人阴险毒辣。
阿朱擅长易容术，能模仿他人相貌声音，最后为救乔峰而死。`;

/**
 * 读取电子书。文件不存在时回落到演示文本。
 * 返回 { text, source, isDemo }
 */
async function loadBook() {
  try {
    const text = await fs.readFile(EBOOK_PATH, "utf8");
    if (!text.trim()) {
      throw new Error(`文件是空的: ${EBOOK_PATH}`);
    }
    return { text, source: EBOOK_PATH, isDemo: false };
  } catch (error) {
    if (error.code === "ENOENT") {
      console.warn(`⚠ 没找到电子书: ${EBOOK_PATH}`);
      console.warn("  改用内置演示文本跑通流程（如需真实数据，见文件头说明）\n");
      return { text: DEMO_TEXT, source: "内置演示文本", isDemo: true };
    }
    throw error;
  }
}

/**
 * 切块。
 */
async function splitIntoChunks(text) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    // 【关键】默认分隔符是英文的段落/句子符号（\n\n、\n、空格、句点），
    // 中文小说几乎不会命中，结果就是按固定字数硬切，经常把句子拦腰截断。
    // 这里按中文标点重新排优先级：段落 → 换行 → 句号 → 感叹/问号 → 分号 → 逗号 → 兜底单字。
    separators: ["\n\n", "\n", "。", "！", "？", "；", "，", ""],
  });

  const docs = await splitter.createDocuments([text]);

  // 切完可能留下只有空格/换行的空块，直接丢掉（空内容算向量没有意义）
  return docs
    .map((doc) => doc.pageContent.trim())
    .filter((content) => content.length > 0);
}

/**
 * 分批把文本变成向量。
 * 一次性把所有块丢给接口会导致请求体过大、超时或被限流。
 */
async function embedInBatches(chunks) {
  const vectors = [];
  for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
    const batch = chunks.slice(i, i + EMBED_BATCH);
    const batchVectors = await embeddings.embedDocuments(batch);
    vectors.push(...batchVectors);
    console.log(`  向量化 ${vectors.length}/${chunks.length}`);
  }
  return vectors;
}

async function main() {
  // ---------- 第 1 步：读文件 ----------
  const { text, source, isDemo } = await loadBook();
  console.log(`书名: ${BOOK_NAME}`);
  console.log(`来源: ${source}`);
  console.log(`字数: ${text.length}`);

  // ---------- 第 2 步：切块 ----------
  console.log(`\n开始切块（每块约 ${CHUNK_SIZE} 字，重叠 ${CHUNK_OVERLAP} 字）...`);
  const chunks = await splitIntoChunks(text);
  console.log(`切出 ${chunks.length} 块`);
  console.log(`第一块预览: ${chunks[0].slice(0, 60)}...`);

  // ---------- 第 3 步：连库、重建集合 ----------
  await client.connectPromise;
  console.log(`\n已连接 Milvus: ${process.env.MILVUS_ADDRESS ?? "localhost:19530"}`);

  // 和 insert.mjs 一样：先删后建，保证脚本可以反复运行。
  const { value: exists } = await client.hasCollection({
    collection_name: COLLECTION_NAME,
  });
  if (exists) {
    await client.dropCollection({ collection_name: COLLECTION_NAME });
    console.log(`集合 ${COLLECTION_NAME} 已存在，已删除并重建`);
  }

  await client.createCollection({
    collection_name: COLLECTION_NAME,
    fields: [
      // 主键：每块一个唯一 id，格式如 book_0001
      { name: "id", data_type: DataType.VarChar, max_length: 64, is_primary_key: true },
      // 向量列
      { name: "vector", data_type: DataType.FloatVector, dim: VECTOR_DIM },
      // 这一块的正文。500 字的块给到 5000 上限很宽松
      { name: "content", data_type: DataType.VarChar, max_length: 5000 },
      // 书名，便于以后同一个集合放多本书
      { name: "book", data_type: DataType.VarChar, max_length: 100 },
      // 这块在原书里的顺序号，检索出来可以按顺序还原上下文
      { name: "chunk_index", data_type: DataType.Int64 },
    ],
  });
  console.log(`集合 ${COLLECTION_NAME} 已创建`);

  // 数据量再小也建议建索引，否则以后查询会退化成全表扫描
  await client.createIndex({
    collection_name: COLLECTION_NAME,
    field_name: "vector",
    index_type: IndexType.FLAT,
    metric_type: MetricType.COSINE,
  });
  await client.loadCollection({ collection_name: COLLECTION_NAME });
  console.log("索引已创建，集合已加载");

  // ---------- 第 4 步：分批向量化 ----------
  console.log("\n开始向量化...");
  const vectors = await embedInBatches(chunks);

  const actualDim = vectors[0]?.length;
  if (actualDim !== VECTOR_DIM) {
    throw new Error(
      `向量维度不匹配：模型返回 ${actualDim} 维，集合定义 ${VECTOR_DIM} 维`,
    );
  }

  // ---------- 第 5 步：分批写入 ----------
  const rows = chunks.map((content, index) => ({
    id: `${BOOK_NAME}_${String(index).padStart(4, "0")}`,
    vector: vectors[index],
    content,
    book: BOOK_NAME,
    chunk_index: index,
  }));

  console.log(`\n开始写入 Milvus（每批 ${INSERT_BATCH} 条）...`);
  let inserted = 0;
  for (let i = 0; i < rows.length; i += INSERT_BATCH) {
    const batch = rows.slice(i, i + INSERT_BATCH);
    const result = await client.insert({
      collection_name: COLLECTION_NAME,
      data: batch,
    });
    inserted += Number(result.insert_cnt);
    console.log(`  已写入 ${inserted}/${rows.length}`);
  }

  // 不 flush 的话，刚写入的数据可能立刻查不到
  await client.flush({ collection_names: [COLLECTION_NAME] });

  console.log(`\n完成：${inserted} 块已写入 ${COLLECTION_NAME}`);
  if (isDemo) {
    console.log("\n⚠ 当前用的是内置演示文本，内容量远小于真书，检索效果有限。");
    console.log("  放入真实电子书后重跑本脚本即可：");
    console.log(`  ${DEFAULT_EBOOK_PATH}`);
  }
}

main()
  .catch((error) => {
    console.error("\n失败:", error.message);
    process.exitCode = 1;
  })
  .finally(async () => {
    await client.closeConnection();
  });
