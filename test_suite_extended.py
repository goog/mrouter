"""扩展测试套件 - 大样本多场景测试"""
from request_classifier_fixed import *
import asyncio
import time
import random
import string
import tracemalloc
from typing import List
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TestCase:
    name: str
    messages: List[dict]
    expected_type: TaskType
    expected_confidence_range: tuple = (0.5, 1.0)
    description: str = ""


class ExtendedTestSuite:
    """扩展测试套件"""
    
    def __init__(self):
        self.classifier = RequestClassifier()
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def get_test_cases(self) -> List[TestCase]:
        """生成扩展测试用例"""
        
        test_cases = []
        
        # ===== CODING 测试 - 15个用例 =====
        coding_tests = [
            ("Python函数实现快速排序", "帮我写一个Python函数实现快速排序", (0.6, 1.0)),
            ("调试bug", "这段代码有bug，帮我debug一下", (0.7, 1.0)),
            ("API设计", "设计一个RESTful API接口", (0.6, 1.0)),
            ("代码重构", "重构这段代码提高可读性", (0.6, 1.0)),
            ("JavaScript异步", "用JavaScript实现异步请求处理", (0.7, 1.0)),
            ("SQL查询优化", "优化这个SQL查询语句的性能", (0.7, 1.0)),
            ("算法实现", "实现一个二叉树遍历算法", (0.7, 1.0)),
            ("代码审查", "Review这段代码并提出改进建议", (0.6, 1.0)),
            ("单元测试", "为这个函数编写单元测试", (0.7, 1.0)),
            ("性能优化", "优化这段代码的执行效率", (0.6, 1.0)),
            ("Docker配置", "写一个Dockerfile配置文件", (0.6, 1.0)),
            ("正则表达式", "写一个正则表达式匹配邮箱", (0.6, 1.0)),
            ("数据结构", "实现一个LRU缓存机制", (0.7, 1.0)),
            ("并发编程", "用多线程实现生产者消费者模式", (0.7, 1.0)),
            ("代码转换", "把这段Java代码转换成Python", (0.7, 1.0)),
        ]
        
        for i, (desc, content, conf_range) in enumerate(coding_tests, 1):
            test_cases.append(TestCase(
                name=f"coding_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=TaskType.CODING,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        # ===== REASONING 测试 - 15个用例 =====
        reasoning_tests = [
            ("数学题分析", "请分析一下这道数学题的解题思路", (0.6, 1.0)),
            ("逻辑证明", "用逻辑推导证明这个命题", (0.7, 1.0)),
            ("分步推理", "Step by step explain how neural networks work", (0.6, 1.0)),
            ("数学计算", "计算这个积分的值并给出推导过程", (0.7, 1.0)),
            ("逻辑推理", "根据前提推导出结论", (0.7, 1.0)),
            ("数据分析", "分析这组数据的统计特征", (0.6, 1.0)),
            ("因果关系", "分析这个现象的因果关系", (0.6, 1.0)),
            ("问题诊断", "诊断这个系统故障的根本原因", (0.6, 1.0)),
            ("方案对比", "对比分析两个方案的优劣", (0.6, 1.0)),
            ("概率推理", "计算这个事件的概率分布", (0.7, 1.0)),
            ("论证分析", "分析这个论证的逻辑结构", (0.7, 1.0)),
            ("算法复杂度", "分析这个算法的时间复杂度", (0.6, 1.0)),
            ("经济分析", "分析这个经济模型的影响因素", (0.6, 1.0)),
            ("科学推理", "基于实验数据推导科学结论", (0.7, 1.0)),
            ("决策分析", "分析这个决策的风险和收益", (0.6, 1.0)),
        ]
        
        for i, (desc, content, conf_range) in enumerate(reasoning_tests, 1):
            test_cases.append(TestCase(
                name=f"reasoning_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=TaskType.REASONING,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        # ===== CREATIVE 测试 - 15个用例 =====
        creative_tests = [
            ("科幻小说", "创作一个关于AI觉醒的科幻短篇小说", (0.6, 1.0)),
            ("诗歌创作", "写一首描写春天的诗", (0.5, 1.0)),
            ("创意头脑风暴", "Brainstorm some creative marketing ideas", (0.6, 1.0)),
            ("故事续写", "续写这个故事的结局", (0.7, 1.0)),
            ("文案创作", "写一个吸引人的广告文案", (0.6, 1.0)),
            ("剧本对话", "创作一段电影剧本对话", (0.7, 1.0)),
            ("歌词创作", "写一首关于青春的歌词", (0.6, 1.0)),
            ("创意写作", "创作一篇寓言故事", (0.7, 1.0)),
            ("角色设计", "设计一个虚构角色的背景故事", (0.6, 1.0)),
            ("世界构建", "构建一个奇幻世界的设定", (0.6, 1.0)),
            ("讽刺文章", "写一篇讽刺现代社会的文章", (0.6, 1.0)),
            ("演讲稿", "写一个激励人心的演讲稿", (0.6, 1.0)),
            ("小说开头", "写一个悬疑小说的开头", (0.7, 1.0)),
            ("创意标题", "为这个产品想10个创意标题", (0.6, 1.0)),
            ("对话创作", "创作两个哲学家之间的对话", (0.6, 1.0)),
        ]
        
        for i, (desc, content, conf_range) in enumerate(creative_tests, 1):
            test_cases.append(TestCase(
                name=f"creative_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=TaskType.CREATIVE,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        # ===== QUICK_QA 测试 - 15个用例 =====
        qa_tests = [
            ("定义解释", "什么是量子计算？请简单解释", (0.6, 1.0)),
            ("文章总结", "Summarize this article briefly", (0.6, 1.0)),
            ("概念介绍", "介绍一下机器学习的基本概念", (0.6, 1.0)),
            ("快速问答", "什么是区块链？", (0.6, 1.0)),
            ("术语解释", "解释一下什么是API", (0.6, 1.0)),
            ("简单说明", "说明一下这个名词的含义", (0.6, 1.0)),
            ("概览请求", "给我一个比特币的overview", (0.6, 1.0)),
            ("简介需求", "简单介绍一下Python语言", (0.6, 1.0)),
            ("定义查询", "define machine learning", (0.6, 1.0)),
            ("总结请求", "总结一下这本书的核心观点", (0.6, 1.0)),
            ("解释说明", "explain the difference between AI and ML", (0.6, 1.0)),
            ("快速了解", "What is deep learning?", (0.6, 1.0)),
            ("概念澄清", "澄清一下这两个概念的区别", (0.6, 1.0)),
            ("简明回答", "简要回答什么是云计算", (0.6, 1.0)),
            ("入门介绍", "给新手介绍一下编程", (0.6, 1.0)),
        ]
        
        for i, (desc, content, conf_range) in enumerate(qa_tests, 1):
            test_cases.append(TestCase(
                name=f"qa_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=TaskType.QUICK_QA,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        # ===== CHINESE 测试 - 15个用例 =====
        chinese_tests = [
            ("日常闲聊", "今天天气怎么样？", (0.8, 1.0)),
            ("问候", "你好，最近怎么样？", (0.8, 1.0)),
            ("推荐请求", "帮我推荐一些好看的中文电影", (0.8, 1.0)),
            ("短文本", "推荐几部电影", (0.8, 1.0)),
            ("建议请求", "给点建议", (0.8, 1.0)),
            ("聊天话题", "聊聊最近的热点新闻", (0.8, 1.0)),
            ("咨询意见", "你觉得这个怎么样？", (0.8, 1.0)),
            ("简单对话", "在吗？", (0.8, 1.0)),
            ("日常问题", "吃了吗？", (0.8, 1.0)),
            ("请求帮助", "帮个忙", (0.8, 1.0)),
            ("意见征求", "你怎么看？", (0.8, 1.0)),
            ("日常交流", "最近在忙什么？", (0.8, 1.0)),
            ("简单请求", "说点什么", (0.8, 1.0)),
            ("聊天", "随便聊聊", (0.8, 1.0)),
            ("中文闲聊", "讲个笑话", (0.8, 1.0)),
        ]
        
        for i, (desc, content, conf_range) in enumerate(chinese_tests, 1):
            test_cases.append(TestCase(
                name=f"chinese_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=TaskType.CHINESE,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        # ===== LONG_CONTEXT 测试 - 5个用例 =====
        long_context_tests = [
            ("超长文本", "这是一个很长的文本..." * 5000, (1.0, 1.0)),
            ("代码仓库", "class Solution:\\n    def solve():\\n        pass\\n" * 2000, (1.0, 1.0)),
            ("长文档", "这是一段很长的文档内容" * 3000, (1.0, 1.0)),
            ("超长对话", "用户: 你好\\n助手: 你好" * 2500, (1.0, 1.0)),
            ("大数据集", "data: [1,2,3,4,5]\\n" * 4000, (1.0, 1.0)),
        ]
        
        for i, (desc, content, conf_range) in enumerate(long_context_tests, 1):
            test_cases.append(TestCase(
                name=f"long_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=TaskType.LONG_CONTEXT,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        # ===== 边界情况测试 - 10个用例 =====
        edge_tests = [
            ("空输入", "", TaskType.GENERAL, (0.3, 0.5)),
            ("单关键词", "代码", TaskType.CODING, (0.5, 1.0)),
            ("中英混合", "用Python实现一个简单的function", TaskType.CODING, (0.7, 1.0)),
            ("多类型混合", "写一段代码并分析它的算法复杂度", TaskType.CODING, (0.5, 1.0)),
            ("关键词在上下文", "我想了解什么是代码注释，请介绍一下", TaskType.QUICK_QA, (0.5, 1.0)),
            ("极短文本", "你好", TaskType.CHINESE, (0.8, 1.0)),
            ("纯英文", "What is the weather today?", TaskType.GENERAL, (0.3, 0.7)),
            ("纯数字", "12345", TaskType.GENERAL, (0.3, 0.6)),
            ("纯标点", "！！！？？", TaskType.GENERAL, (0.3, 0.6)),
            ("重复文本", "测试测试测试测试" * 10, TaskType.CHINESE, (0.8, 1.0)),
        ]
        
        for i, (desc, content, expected, conf_range) in enumerate(edge_tests, 1):
            test_cases.append(TestCase(
                name=f"edge_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=expected,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        # ===== 噪声干扰测试 - 10个用例 =====
        noise_tests = [
            ("噪声干扰-代码", "啊啊啊帮我写个代码吧！！！很简单的那种", TaskType.CODING, (0.5, 1.0)),
            ("噪声干扰-分析", "额...能不能帮我分析一下这个问题？？？谢谢", TaskType.REASONING, (0.5, 1.0)),
            ("噪声干扰-创作", "嘿！帮我创作一个故事！拜托拜托！", TaskType.CREATIVE, (0.5, 1.0)),
            ("噪声干扰-问答", "嗯...什么是AI呢？？？不太懂...", TaskType.QUICK_QA, (0.5, 1.0)),
            ("口语化-代码", "那个...帮我搞个代码呗？就Python的", TaskType.CODING, (0.5, 1.0)),
            ("口语化-创作", "写个好玩的故事呗，嘻嘻", TaskType.CREATIVE, (0.5, 1.0)),
            ("表情符号", "帮我写代码 😊🙏", TaskType.CODING, (0.5, 1.0)),
            ("错别字", "帮我写个代买实现功能", TaskType.CODING, (0.5, 1.0)),
            ("混合语言噪声", "Can you help me 写个代码？Thanks!", TaskType.CODING, (0.5, 1.0)),
            ("重复强调", "写代码写代码写代码！", TaskType.CODING, (0.5, 1.0)),
        ]
        
        for i, (desc, content, expected, conf_range) in enumerate(noise_tests, 1):
            test_cases.append(TestCase(
                name=f"noise_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=expected,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        # ===== 真实场景模拟 - 20个用例 =====
        real_scenario_tests = [
            ("代码求助", "我在写Python爬虫时遇到了SSL错误，能帮我看看吗？", TaskType.CODING, (0.5, 1.0)),
            ("学习咨询", "我想学编程，应该从哪门语言开始？", TaskType.QUICK_QA, (0.5, 1.0)),
            ("技术选型", "我们在做项目，该选React还是Vue？分析一下优劣", TaskType.REASONING, (0.5, 1.0)),
            ("内容创作", "我需要给我的公众号写一篇文章，主题是AI的未来", TaskType.CREATIVE, (0.5, 1.0)),
            ("日常聊天", "周末打算去哪玩？", TaskType.CHINESE, (0.8, 1.0)),
            ("工作问题", "老板让我做个PPT，有没有模板推荐？", TaskType.CHINESE, (0.8, 1.0)),
            ("技术面试", "帮我准备一下算法面试，出一道二叉树的题", TaskType.CODING, (0.5, 1.0)),
            ("产品建议", "分析一下这个产品的用户痛点", TaskType.REASONING, (0.5, 1.0)),
            ("文案需求", "帮我写个产品slogan，要简洁有力", TaskType.CREATIVE, (0.5, 1.0)),
            ("知识学习", "什么是微服务架构？通俗易懂地讲一下", TaskType.QUICK_QA, (0.5, 1.0)),
            ("代码审查", "帮我review一下这段代码，看看有没有bug", TaskType.CODING, (0.5, 1.0)),
            ("数据分析", "分析这份销售数据的趋势和规律", TaskType.REASONING, (0.5, 1.0)),
            ("故事创作", "给我讲个睡前故事吧", TaskType.CREATIVE, (0.5, 1.0)),
            ("概念理解", "解释一下Docker和虚拟机的区别", TaskType.QUICK_QA, (0.5, 1.0)),
            ("生活咨询", "晚上吃什么好呢？", TaskType.CHINESE, (0.8, 1.0)),
            ("性能优化", "我的网站加载很慢，怎么优化？", TaskType.CODING, (0.5, 1.0)),
            ("逻辑推理", "根据这些线索推断凶手是谁", TaskType.REASONING, (0.5, 1.0)),
            ("创意设计", "设计一个有趣的手机APP概念", TaskType.CREATIVE, (0.5, 1.0)),
            ("快速了解", "一句话解释什么是深度学习", TaskType.QUICK_QA, (0.5, 1.0)),
            ("日常问候", "早安！今天也要加油哦", TaskType.CHINESE, (0.8, 1.0)),
        ]
        
        for i, (desc, content, expected, conf_range) in enumerate(real_scenario_tests, 1):
            test_cases.append(TestCase(
                name=f"real_{i:02d}_{desc}",
                messages=[{"role": "user", "content": content}],
                expected_type=expected,
                expected_confidence_range=conf_range,
                description=desc
            ))
        
        return test_cases
    
    async def run_single_test(self, test_case: TestCase) -> dict:
        """运行单个测试"""
        start_time = time.time()
        result = await self.classifier.classify(test_case.messages)
        elapsed = time.time() - start_time
        
        passed = (
            result.task_type == test_case.expected_type and
            test_case.expected_confidence_range[0] <= result.confidence <= test_case.expected_confidence_range[1]
        )
        
        return {
            "name": test_case.name,
            "description": test_case.description,
            "passed": passed,
            "expected_type": test_case.expected_type.value,
            "actual_type": result.task_type.value,
            "confidence": result.confidence,
            "expected_range": test_case.expected_confidence_range,
            "elapsed_ms": elapsed * 1000,
            "detected_lang": result.detected_lang,
            "metadata": result.metadata
        }
    
    async def run_all_tests(self):
        """运行所有测试"""
        test_cases = self.get_test_cases()
        
        print("=" * 80)
        print("扩展测试套件执行报告")
        print("=" * 80)
        print(f"总测试用例数: {len(test_cases)}\n")
        
        for test_case in test_cases:
            result = await self.run_single_test(test_case)
            self.results.append(result)
            
            status = "✓" if result["passed"] else "✗"
            self.passed += 1 if result["passed"] else 0
            self.failed += 1 if not result["passed"] else 0
            
            # 只打印失败的详细信息
            if not result["passed"]:
                print(f"{status} | {result['name']}")
                print(f"   期望: {result['expected_type']} | 实际: {result['actual_type']} | "
                      f"置信度: {result['confidence']:.2f} | 语言: {result['detected_lang']}")
        
        self._print_summary()
    
    def _print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 80)
        print("测试摘要")
        print("=" * 80)
        print(f"总计: {len(self.results)} | ✓ 通过: {self.passed} | ✗ 失败: {self.failed}")
        print(f"通过率: {self.passed / len(self.results) * 100:.1f}%")
        
        # 按类型分组统计
        type_stats = defaultdict(lambda: {"total": 0, "passed": 0})
        for result in self.results:
            expected = result["expected_type"]
            type_stats[expected]["total"] += 1
            if result["passed"]:
                type_stats[expected]["passed"] += 1
        
        print("\n各类型准确率:")
        for task_type, stats in sorted(type_stats.items()):
            accuracy = stats["passed"] / stats["total"] * 100
            bar = "█" * int(accuracy / 5) + "░" * (20 - int(accuracy / 5))
            print(f"  {task_type:15} {stats['passed']:2d}/{stats['total']:2d} "
                  f"[{bar}] {accuracy:5.1f}%")
        
        # 性能统计
        avg_time = sum(r["elapsed_ms"] for r in self.results) / len(self.results)
        max_time = max(r["elapsed_ms"] for r in self.results)
        min_time = min(r["elapsed_ms"] for r in self.results)
        p50 = sorted([r["elapsed_ms"] for r in self.results])[len(self.results) // 2]
        p95 = sorted([r["elapsed_ms"] for r in self.results])[int(len(self.results) * 0.95)]
        
        print(f"\n性能统计:")
        print(f"  平均耗时: {avg_time:.2f}ms")
        print(f"  中位数:   {p50:.2f}ms")
        print(f"  P95:      {p95:.2f}ms")
        print(f"  最大耗时: {max_time:.2f}ms")
        print(f"  最小耗时: {min_time:.2f}ms")


class StressTest:
    """压力测试"""
    
    def __init__(self):
        self.classifier = RequestClassifier()
    
    def _generate_random_text(self, length: int) -> str:
        chars = string.ascii_letters + string.digits + " " + string.punctuation
        return ''.join(random.choice(chars) for _ in range(length))
    
    async def batch_classification_test(self, batch_size: int = 1000):
        """批量分类测试"""
        print(f"\n{'=' * 80}")
        print(f"批量分类测试 - {batch_size} 个随机样本")
        print("=" * 80)
        
        # 生成随机样本
        messages_list = [
            [{"role": "user", "content": self._generate_random_text(random.randint(50, 500))}]
            for _ in range(batch_size)
        ]
        
        # 执行批量分类
        start_time = time.time()
        results = []
        for messages in messages_list:
            result = await self.classifier.classify(messages)
            results.append(result)
        total_time = time.time() - start_time
        
        # 统计结果
        type_dist = defaultdict(int)
        lang_dist = defaultdict(int)
        conf_dist = {"low": 0, "medium": 0, "high": 0}
        
        for result in results:
            type_dist[result.task_type.value] += 1
            lang_dist[result.detected_lang] += 1
            if result.confidence < 0.5:
                conf_dist["low"] += 1
            elif result.confidence < 0.8:
                conf_dist["medium"] += 1
            else:
                conf_dist["high"] += 1
        
        print(f"\n吞吐量: {batch_size / total_time:.1f} req/s")
        print(f"平均延迟: {total_time / batch_size * 1000:.2f}ms")
        
        print(f"\n任务类型分布:")
        for task_type, count in sorted(type_dist.items(), key=lambda x: -x[1]):
            pct = count / batch_size * 100
            print(f"  {task_type:15} {count:4d} ({pct:5.1f}%)")
        
        print(f"\n语言分布:")
        for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1])[:5]:
            pct = count / batch_size * 100
            print(f"  {lang:10} {count:4d} ({pct:5.1f}%)")
        
        print(f"\n置信度分布:")
        for level, count in conf_dist.items():
            pct = count / batch_size * 100
            print(f"  {level:10} {count:4d} ({pct:5.1f}%)")
        
        return {
            "throughput": batch_size / total_time,
            "avg_latency": total_time / batch_size * 1000,
            "type_distribution": dict(type_dist),
            "lang_distribution": dict(lang_dist),
            "conf_distribution": conf_dist
        }
    
    async def concurrent_stress_test(self, concurrent_count: int = 500):
        """并发压力测试"""
        print(f"\n{'=' * 80}")
        print(f"并发压力测试 - {concurrent_count} 个并发请求")
        print("=" * 80)
        
        messages_list = [
            [{"role": "user", "content": self._generate_random_text(random.randint(100, 300))}]
            for _ in range(concurrent_count)
        ]
        
        start_time = time.time()
        tasks = [
            self.classifier.classify(messages)
            for messages in messages_list
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"\n并发吞吐量: {concurrent_count / total_time:.1f} req/s")
        print(f"总耗时: {total_time * 1000:.2f}ms")
        print(f"平均单请求: {total_time / concurrent_count * 1000:.2f}ms")
        
        return {
            "throughput": concurrent_count / total_time,
            "total_time_ms": total_time * 1000,
            "avg_per_request_ms": total_time / concurrent_count * 1000
        }


async def main():
    """运行完整测试"""
    # 1. 扩展测试套件
    test_suite = ExtendedTestSuite()
    await test_suite.run_all_tests()
    
    # 2. 压力测试
    stress_test = StressTest()
    await stress_test.batch_classification_test(batch_size=1000)
    await stress_test.concurrent_stress_test(concurrent_count=500)


if __name__ == "__main__":
    asyncio.run(main())
