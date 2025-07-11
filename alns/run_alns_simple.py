#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版ALNS运行程序，用于快速测试
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_simple_alns():
    """运行简化版ALNS"""
    print("=== 启动简化版ALNS算法 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 导入必要模块
        print("1. 导入模块...")
        from data_loader import load_all_data
        from unified_config import UnifiedConfig
        from initial_solution_generator import generate_initial_rosters_with_heuristic
        from main import ALNSSolution, ALNSAlgorithm
        print("✅ 模块导入成功")
        
        # 2. 加载数据（使用较小的数据集）
        print("2. 加载数据...")
        data_path = UnifiedConfig.DATA_PATH
        all_data = load_all_data(data_path)
        
        if not all_data:
            print("❌ 数据加载失败")
            return False
        
        # 使用较小的数据集进行测试
        flights = all_data["flights"][:50]  # 50个航班
        crews = all_data["crews"][:20]      # 20个机组
        ground_duties = all_data["ground_duties"][:20]
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
        
        print(f"✅ 数据加载完成: 航班{len(flights)}个, 机组{len(crews)}个")
        
        # 3. 生成初始解
        print("3. 生成初始解...")
        initial_rosters = generate_initial_rosters_with_heuristic(
            flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
        )
        
        if not initial_rosters:
            print("❌ 初始解生成失败")
            return False
        
        initial_solution = ALNSSolution(initial_rosters, flights, ground_duties, crews)
        print(f"✅ 初始解生成完成: {initial_solution}")
        
        # 4. 创建ALNS算法实例
        print("4. 创建ALNS算法实例...")
        alns = ALNSAlgorithm(flights, crews, ground_duties, bus_info,
                            crew_leg_match_dict, layover_stations)
        
        # 设置较短的运行时间进行测试
        alns.time_limit = 300  # 5分钟测试
        
        print(f"算法配置:")
        print(f"  时间限制: {alns.time_limit}秒")
        print(f"  破坏算子: {[op.name for op in alns.destroy_operators]}")
        print(f"  修复算子: {[op.name for op in alns.repair_operators]}")
        
        # 5. 运行ALNS算法
        print("5. 开始运行ALNS算法...")
        start_time = datetime.now()
        
        best_solution = alns.solve(initial_solution)
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # 6. 输出结果
        print(f"\n=== ALNS运行结果 ===")
        print(f"运行时间: {runtime:.2f}秒")
        print(f"总迭代次数: {alns.iteration_count + 1}")
        print(f"平均每秒迭代: {(alns.iteration_count + 1) / runtime:.2f}")
        
        print(f"\n解质量对比:")
        print(f"  初始解目标值: {initial_solution.objective_value:.2f}")
        print(f"  最终解目标值: {best_solution.objective_value:.2f}")
        improvement = initial_solution.objective_value - best_solution.objective_value
        print(f"  改进: {improvement:.2f}")
        
        print(f"\n覆盖率对比:")
        print(f"  初始解覆盖率: {initial_solution.coverage_rate:.2%}")
        print(f"  最终解覆盖率: {best_solution.coverage_rate:.2%}")
        
        print(f"\nRoster数量:")
        print(f"  初始解: {len(initial_solution.rosters)}个")
        print(f"  最终解: {len(best_solution.rosters)}个")
        
        # 7. 算子使用统计
        print(f"\n=== 算子使用统计 ===")
        for op in alns.destroy_operators:
            success_rate = op.success_count / op.usage_count if op.usage_count > 0 else 0
            print(f"破坏算子 {op.name}: 使用{op.usage_count}次, 成功{op.success_count}次, 成功率{success_rate:.2%}")
        
        for op in alns.repair_operators:
            success_rate = op.success_count / op.usage_count if op.usage_count > 0 else 0
            print(f"修复算子 {op.name}: 使用{op.usage_count}次, 成功{op.success_count}次, 成功率{success_rate:.2%}")
        
        # 判断运行效果
        if improvement > 0:
            print(f"\n✅ ALNS算法运行成功！找到了更好的解决方案。")
        elif improvement == 0:
            print(f"\n⚠️  ALNS算法运行完成，但没有找到改进。")
        else:
            print(f"\n❌ ALNS算法运行完成，但解质量有所下降。")
        
        print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("启动简化版ALNS算法测试...")
    
    success = run_simple_alns()
    
    if success:
        print("\n🎉 简化版ALNS测试完成！")
    else:
        print("\n💥 简化版ALNS测试失败！")
