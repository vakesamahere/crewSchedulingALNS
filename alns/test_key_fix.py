#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试关键修复：确保被移除的任务能被重新分配
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
from main import ALNSSolution, ALNSAlgorithm, WorstRosterDestroy, GurobiRepair

def test_key_fix():
    """测试关键修复效果"""
    print("=== 测试关键修复：被移除任务的重新分配 ===")
    
    try:
        # 1. 加载小数据集
        print("1. 加载数据...")
        data_path = UnifiedConfig.DATA_PATH
        all_data = load_all_data(data_path)
        
        flights = all_data["flights"][:20]  # 20个航班
        crews = all_data["crews"][:10]      # 10个机组
        ground_duties = all_data["ground_duties"][:10]
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
        
        # 2. 生成初始解
        print("2. 生成初始解...")
        initial_rosters = generate_initial_rosters_with_heuristic(
            flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
        )
        
        if not initial_rosters:
            print("初始解生成失败")
            return False
        
        initial_solution = ALNSSolution(initial_rosters, flights, ground_duties, crews)
        print(f"初始解: {initial_solution}")
        
        # 3. 手动测试破坏-修复循环
        print("3. 测试破坏-修复循环...")
        
        # 创建算子
        destroy_op = WorstRosterDestroy()
        repair_op = GurobiRepair(crews, flights, ground_duties, bus_info,
                               crew_leg_match_dict, layover_stations)
        
        # 执行破坏操作
        print("执行破坏操作...")
        destroyed_solution, removed_rosters = destroy_op.destroy(initial_solution, 2)
        
        print(f"破坏前: {len(initial_solution.rosters)}个rosters")
        print(f"破坏后: {len(destroyed_solution.rosters)}个rosters")
        print(f"移除了: {len(removed_rosters)}个rosters")
        
        # 统计被移除的任务
        removed_tasks = []
        for roster in removed_rosters:
            removed_tasks.extend(roster.duties)
        
        print(f"被移除的任务数量: {len(removed_tasks)}")
        for i, task in enumerate(removed_tasks):
            task_type = "航班" if hasattr(task, 'depaAirport') else "其他"
            task_id = getattr(task, 'id', f'task_{i}')
            print(f"  - {task_type}: {task_id}")
        
        # 执行修复操作
        print("执行修复操作...")
        repaired_solution = repair_op.repair(destroyed_solution, removed_rosters)
        
        print(f"修复后: {len(repaired_solution.rosters)}个rosters")
        
        # 4. 分析修复效果
        print("4. 分析修复效果...")
        
        # 统计修复后的任务
        repaired_tasks = []
        for roster in repaired_solution.rosters:
            repaired_tasks.extend(roster.duties)
        
        # 检查被移除的任务是否被重新分配
        removed_task_ids = {getattr(task, 'id', str(task)) for task in removed_tasks}
        repaired_task_ids = {getattr(task, 'id', str(task)) for task in repaired_tasks}
        
        recovered_tasks = removed_task_ids & repaired_task_ids
        lost_tasks = removed_task_ids - repaired_task_ids
        
        print(f"被移除任务数量: {len(removed_task_ids)}")
        print(f"成功恢复任务数量: {len(recovered_tasks)}")
        print(f"丢失任务数量: {len(lost_tasks)}")
        print(f"恢复率: {len(recovered_tasks)/len(removed_task_ids)*100:.1f}%")
        
        if lost_tasks:
            print("丢失的任务:")
            for task_id in lost_tasks:
                print(f"  - {task_id}")
        
        # 5. 比较目标函数
        print("5. 目标函数比较...")
        print(f"初始解目标值: {initial_solution.objective_value:.2f}")
        print(f"破坏后目标值: {destroyed_solution.objective_value:.2f}")
        print(f"修复后目标值: {repaired_solution.objective_value:.2f}")
        
        improvement = initial_solution.objective_value - repaired_solution.objective_value
        print(f"总体改进: {improvement:.2f}")
        
        # 判断修复效果
        if len(recovered_tasks) >= len(removed_task_ids) * 0.8:  # 80%恢复率
            print("✅ 关键修复成功！大部分被移除的任务得到了重新分配。")
            return True
        else:
            print("⚠️  关键修复部分成功，但仍有改进空间。")
            return False
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试关键修复...")
    
    success = test_key_fix()
    
    print(f"\n=== 测试结果 ===")
    if success:
        print("✅ 关键修复测试成功！")
        print("现在被移除的任务可以被重新分配，这应该能显著改善ALNS的优化效果。")
    else:
        print("❌ 关键修复需要进一步调整。")
