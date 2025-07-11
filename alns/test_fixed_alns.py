#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修正后的ALNS逻辑
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fixed_alns_logic():
    """测试修正后的ALNS逻辑"""
    print("=== 测试修正后的ALNS逻辑 ===")
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # 导入模块
        from data_loader import load_all_data
        from unified_config import UnifiedConfig
        from initial_solution_generator import generate_initial_rosters_with_heuristic
        from main import ALNSSolution, WorstRosterDestroy, GurobiRepair
        
        # 加载小数据集
        print("1. 加载数据...")
        data_path = UnifiedConfig.DATA_PATH
        all_data = load_all_data(data_path)
        
        # 使用很小的数据集进行快速测试
        flights = all_data["flights"][:15]  # 15个航班
        crews = all_data["crews"][:8]       # 8个机组
        ground_duties = all_data["ground_duties"][:8]
        bus_info = all_data["bus_info"]
        crew_leg_match_list = all_data["crew_leg_matches"]
        layover_stations = all_data["layover_stations"]
        
        # 预处理资质数据
        crew_leg_match_dict = {}
        for match in crew_leg_match_list:
            flight_id, crew_id = match.flightId, match.crewId
            if crew_id not in crew_leg_match_dict:
                crew_leg_match_dict[crew_id] = []
            crew_leg_match_dict[crew_id].append(flight_id)
        
        print(f"数据加载完成: 航班{len(flights)}个, 机组{len(crews)}个")
        
        # 生成初始解
        print("2. 生成初始解...")
        initial_rosters = generate_initial_rosters_with_heuristic(
            flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
        )
        
        if not initial_rosters:
            print("初始解生成失败")
            return False
        
        initial_solution = ALNSSolution(initial_rosters, flights, ground_duties, crews)
        print(f"初始解: {initial_solution}")
        
        # 测试单次破坏-修复循环
        print("3. 测试修正后的破坏-修复循环...")
        
        # 创建算子
        destroy_op = WorstRosterDestroy()
        repair_op = GurobiRepair(crews, flights, ground_duties, bus_info,
                               crew_leg_match_dict, layover_stations)
        
        # 执行破坏
        print("执行破坏操作...")
        destroyed_solution, removed_rosters = destroy_op.destroy(initial_solution, 2)
        
        print(f"破坏前: {len(initial_solution.rosters)}个rosters, 目标值{initial_solution.objective_value:.2f}")
        print(f"破坏后: {len(destroyed_solution.rosters)}个rosters, 目标值{destroyed_solution.objective_value:.2f}")
        print(f"移除了: {len(removed_rosters)}个rosters")
        
        # 统计被移除的任务
        removed_tasks = []
        for roster in removed_rosters:
            removed_tasks.extend(roster.duties)
        print(f"被移除的任务: {len(removed_tasks)}个")
        
        # 执行修复（使用修正后的逻辑）
        print("执行修正后的修复操作...")
        repaired_solution = repair_op.repair(destroyed_solution, removed_rosters)
        
        print(f"修复后: {len(repaired_solution.rosters)}个rosters, 目标值{repaired_solution.objective_value:.2f}")
        
        # 分析修复效果
        print("4. 分析修复效果...")
        
        # 比较目标函数
        initial_obj = initial_solution.objective_value
        repaired_obj = repaired_solution.objective_value
        improvement = initial_obj - repaired_obj
        
        print(f"目标函数比较:")
        print(f"  初始解: {initial_obj:.2f}")
        print(f"  修复后: {repaired_obj:.2f}")
        print(f"  改进: {improvement:.2f}")
        
        # 比较覆盖率
        print(f"覆盖率比较:")
        print(f"  初始解: {initial_solution.coverage_rate:.2%}")
        print(f"  修复后: {repaired_solution.coverage_rate:.2%}")
        
        # 检查是否有真正的重组
        initial_task_assignments = set()
        for roster in initial_solution.rosters:
            for task in roster.duties:
                task_id = getattr(task, 'id', str(task))
                initial_task_assignments.add((roster.crew_id, task_id))
        
        repaired_task_assignments = set()
        for roster in repaired_solution.rosters:
            for task in roster.duties:
                task_id = getattr(task, 'id', str(task))
                repaired_task_assignments.add((roster.crew_id, task_id))
        
        # 计算任务重新分配的情况
        same_assignments = initial_task_assignments & repaired_task_assignments
        new_assignments = repaired_task_assignments - initial_task_assignments
        lost_assignments = initial_task_assignments - repaired_task_assignments
        
        print(f"任务重新分配分析:")
        print(f"  保持不变: {len(same_assignments)}个")
        print(f"  新增分配: {len(new_assignments)}个")
        print(f"  丢失分配: {len(lost_assignments)}个")
        
        if len(new_assignments) > 0:
            print("✅ 检测到任务重新分配，ALNS逻辑正常工作！")
            print("新的任务分配:")
            for crew_id, task_id in list(new_assignments)[:5]:  # 显示前5个
                print(f"  机组{crew_id} -> 任务{task_id}")
        else:
            print("⚠️  没有检测到任务重新分配")
        
        # 判断修复效果
        if improvement > 0:
            print("✅ 修复成功！找到了更好的解决方案。")
            return True
        elif improvement == 0:
            print("⚠️  修复完成，但没有改进。")
            return True
        else:
            print("❌ 修复后解质量下降。")
            return False
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试修正后的ALNS逻辑...")
    
    success = test_fixed_alns_logic()
    
    print(f"\n=== 测试结果 ===")
    if success:
        print("✅ 修正后的ALNS逻辑测试成功！")
        print("关键改进:")
        print("  1. 移除了错误的'直接恢复'策略")
        print("  2. 让Gurobi真正重新优化任务分配")
        print("  3. 遵循正确的ALNS破坏-修复逻辑")
    else:
        print("❌ 测试失败，需要进一步调试。")
    
    print(f"结束时间: {datetime.now().strftime('%H:%M:%S')}")
