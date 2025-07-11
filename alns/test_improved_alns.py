#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试改进后的ALNS算法
"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_all_data
from unified_config import UnifiedConfig
from initial_solution_generator import generate_initial_rosters_with_heuristic

# 导入ALNS相关类
from main import ALNSSolution, ALNSAlgorithm

def test_improved_alns():
    """测试改进后的ALNS算法"""
    print("=== 测试改进后的ALNS算法 ===")
    
    try:
        # 1. 加载数据
        print("1. 加载数据...")
        data_path = UnifiedConfig.DATA_PATH
        all_data = load_all_data(data_path)
        
        if not all_data:
            print("数据加载失败")
            return False
        
        # 使用中等规模数据集
        flights = all_data["flights"][:100]  # 100个航班
        crews = all_data["crews"][:30]       # 30个机组
        ground_duties = all_data["ground_duties"][:30]
        bus_info = all_data["bus_info"]
        crew_leg_match_list = all_data["crew_leg_matches"]
        layover_stations = all_data["layover_stations"]
        
        # 预处理机长-航班资质数据
        crew_leg_match_dict = {}
        for match in crew_leg_match_list:
            flight_id, crew_id = match.flightId, match.crewId
            if crew_id not in crew_leg_match_dict:
                crew_leg_match_dict[crew_id] = []
            crew_leg_match_dict[crew_id].append(flight_id)
        
        print(f"数据加载完成: 航班{len(flights)}个, 机组{len(crews)}个")
        
        # 2. 生成初始解
        print("2. 生成初始解...")
        initial_rosters = generate_initial_rosters_with_heuristic(
            flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
        )
        
        if not initial_rosters:
            print("初始解生成失败")
            return False
        
        print(f"初始解生成完成: {len(initial_rosters)}个rosters")
        
        # 3. 创建ALNS解决方案对象
        print("3. 创建ALNS解决方案...")
        initial_solution = ALNSSolution(initial_rosters, flights, ground_duties, crews)
        print(f"初始解: {initial_solution}")
        
        # 4. 创建ALNS算法实例
        print("4. 创建改进的ALNS算法实例...")
        alns = ALNSAlgorithm(flights, crews, ground_duties, bus_info,
                            crew_leg_match_dict, layover_stations)
        
        # 验证改进的参数
        print(f"改进的参数设置:")
        print(f"  破坏大小范围: {alns.destroy_size_min} - {alns.destroy_size_max}")
        print(f"  初始温度: {alns.initial_temperature}")
        print(f"  冷却速度: {alns.cooling_rate}")
        print(f"  最小温度: {alns.min_temperature}")
        print(f"  停滞限制: {alns.stagnation_limit}")
        
        # 5. 设置测试运行时间
        alns.time_limit = 120  # 2分钟测试
        
        print(f"5. 执行改进的ALNS算法（{alns.time_limit}秒）...")
        start_time = datetime.now()
        
        # 执行算法
        best_solution = alns.solve(initial_solution)
        
        end_time = datetime.now()
        actual_runtime = (end_time - start_time).total_seconds()
        
        # 6. 输出结果
        print(f"\n=== 改进效果分析 ===")
        print(f"实际运行时间: {actual_runtime:.2f}秒")
        print(f"总迭代次数: {alns.iteration_count + 1}")
        print(f"平均每秒迭代次数: {(alns.iteration_count + 1) / actual_runtime:.2f}")
        
        print(f"\n解质量对比:")
        print(f"  初始解目标值: {initial_solution.objective_value:.2f}")
        print(f"  最终解目标值: {best_solution.objective_value:.2f}")
        improvement = initial_solution.objective_value - best_solution.objective_value
        improvement_pct = (improvement / abs(initial_solution.objective_value)) * 100
        print(f"  绝对改进: {improvement:.2f}")
        print(f"  相对改进: {improvement_pct:.2f}%")
        
        print(f"\n覆盖率对比:")
        print(f"  初始解覆盖率: {initial_solution.coverage_rate:.2%}")
        print(f"  最终解覆盖率: {best_solution.coverage_rate:.2%}")
        coverage_improvement = best_solution.coverage_rate - initial_solution.coverage_rate
        print(f"  覆盖率改进: {coverage_improvement:.2%}")
        
        print(f"\nRoster数量对比:")
        print(f"  初始解rosters: {len(initial_solution.rosters)}")
        print(f"  最终解rosters: {len(best_solution.rosters)}")
        
        # 7. 多样化策略效果
        print(f"\n=== 多样化策略效果 ===")
        print(f"最后改进迭代: {alns.last_improvement_iteration}")
        print(f"当前多样化因子: {alns.diversification_factor:.2f}")
        print(f"最终温度: {alns.temperature:.2f}")
        
        # 8. 算子使用统计
        print(f"\n=== 算子使用统计 ===")
        for op in alns.destroy_operators:
            success_rate = op.success_count / op.usage_count if op.usage_count > 0 else 0
            print(f"破坏算子 {op.name}: 使用{op.usage_count}次, 成功{op.success_count}次, 成功率{success_rate:.2%}")
        
        for op in alns.repair_operators:
            success_rate = op.success_count / op.usage_count if op.usage_count > 0 else 0
            print(f"修复算子 {op.name}: 使用{op.usage_count}次, 成功{op.success_count}次, 成功率{success_rate:.2%}")
        
        # 判断改进效果
        if improvement > 0 or coverage_improvement > 0:
            print(f"\n✅ 改进成功！算法找到了更好的解决方案。")
            return True
        else:
            print(f"\n⚠️  改进有限，可能需要进一步调整参数。")
            return False
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试改进后的ALNS算法...")
    
    success = test_improved_alns()
    
    print(f"\n=== 最终测试结果 ===")
    if success:
        print("✅ 改进的ALNS算法测试成功！")
        print("主要改进包括:")
        print("  1. 更智能的WorstRosterDestroy（基于成本效益比）")
        print("  2. 增加的破坏大小范围（2-10个rosters）")
        print("  3. 更慢的模拟退火冷却速度")
        print("  4. 动态多样化策略")
        print("  5. 改进的候选任务选择")
        print("  6. 优化的Gurobi求解参数")
        print("\n现在可以运行完整的1小时优化！")
    else:
        print("❌ 测试未达到预期效果，可能需要进一步调整。")
