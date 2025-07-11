#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALNS (Adaptive Large Neighborhood Search) 框架
用于机组排班优化问题

基于现有的约束检查器、覆盖验证器和初始解生成器
实现自适应大邻域搜索算法

Author: Crew Scheduling Team
Date: 2025-01-10
"""

import os
import sys
import time
import random
import math
import copy
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Optional, Any
from gurobi_repair_solver import GurobiRepairSolver, RepairInput, RepairResult

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from constraint_checker import UnifiedConstraintChecker
from coverage_validator import CoverageValidator
from initial_solution_generator import generate_initial_rosters_with_heuristic
from data_loader import load_all_data
from data_models import Flight, Crew, Roster, GroundDuty, BusInfo
# from results_writer import write_results_to_csv  # 使用自定义的简化版本
from unified_config import UnifiedConfig


class ALNSSolution:
    """ALNS解决方案类"""

    def __init__(self, rosters: List[Roster], flights: List[Flight],
                 ground_duties: List[GroundDuty], crews: List[Crew]):
        self.rosters = rosters
        self.flights = flights
        self.ground_duties = ground_duties
        self.crews = crews
        self.objective_value = None
        self.coverage_rate = None
        self.ground_duty_coverage_rate = None
        self.is_feasible = True
        self.violations = []
        # 计算目标函数值
        self._calculate_objective()


    def _calculate_objective(self):
            """计算目标函数值"""
            # 使用与main.py相同的线性目标函数
            total_flight_hours = 0.0
            total_duty_days = 0.0
            covered_flights = set()
            covered_ground_duties = set()

            roster_cost_sum = 0
            for roster in self.rosters:
                # 计算roster成本（使用统一配置的参数）
                flight_reward = 0
                positioning_penalty = 0
                overnight_penalty = 0

                for duty in roster.duties:
                    if isinstance(duty, Flight):
                        covered_flights.add(duty.id)
                        if hasattr(duty, 'flyTime') and duty.flyTime:
                            flight_time_hours = duty.flyTime / 60.0
                            flight_reward += flight_time_hours * UnifiedConfig.FLIGHT_TIME_REWARD
                            total_flight_hours += flight_time_hours
                        total_duty_days += 1
                    elif hasattr(duty, 'crewId') and hasattr(duty, 'airport'):
                        covered_ground_duties.add(duty.id)
                        total_duty_days += 1

                # 简化的成本计算（主要组成部分）
                roster_cost = flight_reward - positioning_penalty - overnight_penalty
                roster_cost_sum += roster_cost

            # 计算未覆盖惩罚
            uncovered_flights = len(self.flights) - len(covered_flights)
            uncovered_ground_duties = len(self.ground_duties) - len(covered_ground_duties)

            uncovered_flight_penalty = uncovered_flights * UnifiedConfig.UNCOVERED_FLIGHT_PENALTY
            uncovered_ground_duty_penalty = uncovered_ground_duties * UnifiedConfig.UNCOVERED_GROUND_DUTY_PENALTY

            # 总目标函数值（最小化）
            self.objective_value = roster_cost_sum + uncovered_flight_penalty + uncovered_ground_duty_penalty

            # 计算覆盖率
            self.coverage_rate = len(covered_flights) / len(self.flights) if self.flights else 0.0
            self.ground_duty_coverage_rate = len(covered_ground_duties) / len(self.ground_duties) if self.ground_duties else 0.0

    def copy(self):
        """创建解的深拷贝"""
        new_rosters = [copy.deepcopy(roster) for roster in self.rosters]
        return ALNSSolution(new_rosters, self.flights, self.ground_duties, self.crews)

    def is_better_than(self, other_solution):
        """判断当前解是否优于另一个解"""
        if not self.is_feasible and other_solution.is_feasible:
            return False
        if self.is_feasible and not other_solution.is_feasible:
            return True
        return self.objective_value < other_solution.objective_value

    def __str__(self):
        return (f"ALNSSolution(obj={self.objective_value:.2f}, "
                f"coverage={self.coverage_rate:.2%}, "
                f"rosters={len(self.rosters)}, "
                f"feasible={self.is_feasible})")

    
class DestroyOperator:
    """破坏算子基类"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.usage_count = 0
        self.success_count = 0

    def destroy(self, solution: ALNSSolution, destroy_size: int) -> Tuple[ALNSSolution, List[Any]]:
        """
        破坏解决方案

        Args:
            solution: 当前解决方案
            destroy_size: 破坏的大小（移除的元素数量）

        Returns:
            Tuple[破坏后的解决方案, 被移除的元素列表]
        """
        raise NotImplementedError

    def update_weight(self, success: bool):
        """更新算子权重"""
        self.usage_count += 1
        if success:
            self.success_count += 1


class RepairOperator:
    """修复算子基类"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.usage_count = 0
        self.success_count = 0

    def repair(self, solution: ALNSSolution, removed_elements: List[Any]) -> ALNSSolution:
        """
        修复解决方案

        Args:
            solution: 被破坏的解决方案
            removed_elements: 被移除的元素列表

        Returns:
            修复后的解决方案
        """
        raise NotImplementedError

    def update_weight(self, success: bool):
        """更新算子权重"""
        self.usage_count += 1
        if success:
            self.success_count += 1


class RandomRosterDestroy(DestroyOperator):
    """随机移除roster的破坏算子"""

    def __init__(self):
        super().__init__("RandomRosterDestroy")

    def destroy(self, solution: ALNSSolution, destroy_size: int) -> Tuple[ALNSSolution, List[Roster]]:
        """随机移除指定数量的roster"""
        new_solution = solution.copy()

        if len(new_solution.rosters) <= destroy_size:
            # 如果要移除的数量大于等于总数，保留一个roster
            destroy_size = max(1, len(new_solution.rosters) - 1)

        # 随机选择要移除的roster
        removed_rosters = random.sample(new_solution.rosters, destroy_size)

        # 从解中移除选中的roster
        for roster in removed_rosters:
            new_solution.rosters.remove(roster)

        # 重新计算目标函数
        new_solution._calculate_objective()

        return new_solution, removed_rosters


class WorstRosterDestroy(DestroyOperator):
    """移除最差roster的破坏算子"""

    def __init__(self):
        super().__init__("WorstRosterDestroy")

    def destroy(self, solution: ALNSSolution, destroy_size: int) -> Tuple[ALNSSolution, List[Roster]]:
        """移除成本效益比最差的roster"""
        new_solution = solution.copy()

        if len(new_solution.rosters) <= destroy_size:
            destroy_size = max(1, len(new_solution.rosters) - 1)

        # 计算每个roster的成本效益比
        roster_scores = []
        for roster in new_solution.rosters:
            # 计算roster的负面影响：成本高、覆盖任务少、违规多
            cost = getattr(roster, 'cost', 0)
            task_count = len(roster.duties)
            flight_count = sum(1 for duty in roster.duties if isinstance(duty, Flight))

            # 成本效益比：成本越高、覆盖任务越少，分数越高（越差）
            if task_count > 0:
                cost_per_task = cost / task_count
                # 如果没有航班任务，额外惩罚
                if flight_count == 0:
                    cost_per_task += 1000
            else:
                cost_per_task = float('inf')

            roster_scores.append((cost_per_task, roster))

        # 按成本效益比排序，选择最差的
        roster_scores.sort(key=lambda x: x[0], reverse=True)
        removed_rosters = [roster for _, roster in roster_scores[:destroy_size]]

        # 从解中移除选中的roster
        for roster in removed_rosters:
            new_solution.rosters.remove(roster)

        new_solution._calculate_objective()

        return new_solution, removed_rosters


class RelatedFlightDestroy(DestroyOperator):
    """移除相关航班的破坏算子"""

    def __init__(self):
        super().__init__("RelatedFlightDestroy")

    def destroy(self, solution: ALNSSolution, destroy_size: int) -> Tuple[ALNSSolution, List[Roster]]:
        """移除包含相关航班的roster"""
        new_solution = solution.copy()

        if not new_solution.rosters:
            return new_solution, []

        # 随机选择一个起始航班
        all_flights_in_rosters = []
        for roster in new_solution.rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    all_flights_in_rosters.append((duty, roster))

        if not all_flights_in_rosters:
            # 如果没有航班，回退到随机移除
            return RandomRosterDestroy().destroy(solution, destroy_size)

        seed_flight, _ = random.choice(all_flights_in_rosters)

        # 找到相关的航班（相同机场或相近时间）
        related_rosters = set()
        for roster in new_solution.rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    # 检查是否为相关航班
                    if (duty.depaAirport == seed_flight.depaAirport or
                        duty.arriAirport == seed_flight.arriAirport or
                        abs((duty.std - seed_flight.std).total_seconds()) < 3600):  # 1小时内
                        related_rosters.add(roster)
                        break

        # 限制移除数量
        related_rosters = list(related_rosters)
        if len(related_rosters) > destroy_size:
            related_rosters = random.sample(related_rosters, destroy_size)
        elif len(related_rosters) == 0:
            # 如果没有找到相关roster，随机选择
            related_rosters = random.sample(new_solution.rosters,
                                          min(destroy_size, len(new_solution.rosters)))

        # 移除选中的roster
        for roster in related_rosters:
            if roster in new_solution.rosters:
                new_solution.rosters.remove(roster)

        new_solution._calculate_objective()

        return new_solution, related_rosters


class GreedyRepair(RepairOperator):
    """贪心修复算子"""

    def __init__(self, crews: List[Crew], flights: List[Flight],
                 ground_duties: List[GroundDuty], bus_info: List[BusInfo],
                 crew_leg_match_dict: Dict, layover_stations: Set[str]):
        super().__init__("GreedyRepair")
        self.crews = crews
        self.flights = flights
        self.ground_duties = ground_duties
        self.bus_info = bus_info
        self.crew_leg_match_dict = crew_leg_match_dict
        self.layover_stations = layover_stations
        self.constraint_checker = UnifiedConstraintChecker(layover_stations)

    def repair(self, solution: ALNSSolution, removed_rosters: List[Roster]) -> ALNSSolution:
        """使用贪心策略修复解决方案"""
        new_solution = solution.copy()

        # 获取当前未覆盖的航班和地面任务
        covered_flights = set()
        covered_ground_duties = set()

        for roster in new_solution.rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    covered_flights.add(duty.id)
                elif hasattr(duty, 'crewId') and hasattr(duty, 'airport'):
                    covered_ground_duties.add(duty.id)

        uncovered_flights = [f for f in self.flights if f.id not in covered_flights]
        uncovered_ground_duties = [gd for gd in self.ground_duties if gd.id not in covered_ground_duties]

        # 获取可用的机组（没有被分配roster的机组）
        assigned_crews = {roster.crew_id for roster in new_solution.rosters}
        available_crews = [crew for crew in self.crews if crew.crewId not in assigned_crews]

        # 为每个可用机组尝试创建新的roster
        for crew in available_crews:
            if not uncovered_flights and not uncovered_ground_duties:
                break

            # 获取该机组可执行的航班
            eligible_flight_ids = self.crew_leg_match_dict.get(crew.crewId, [])
            eligible_flights = [f for f in uncovered_flights if f.id in eligible_flight_ids]

            # 获取该机组的地面任务
            crew_ground_duties = [gd for gd in uncovered_ground_duties if gd.crewId == crew.crewId]

            # 尝试创建一个简单的roster
            new_roster = self._create_simple_roster(crew, eligible_flights, crew_ground_duties)

            if new_roster and new_roster.duties:
                new_solution.rosters.append(new_roster)

                # 更新未覆盖列表
                for duty in new_roster.duties:
                    if isinstance(duty, Flight) and duty in uncovered_flights:
                        uncovered_flights.remove(duty)
                    elif hasattr(duty, 'crewId') and duty in uncovered_ground_duties:
                        uncovered_ground_duties.remove(duty)

        new_solution._calculate_objective()
        return new_solution

    def _create_simple_roster(self, crew: Crew, eligible_flights: List[Flight],
                             crew_ground_duties: List[GroundDuty]) -> Optional[Roster]:
        """为机组创建一个简单的roster"""
        if not eligible_flights and not crew_ground_duties:
            return None

        # 合并所有任务并按时间排序
        all_tasks = []
        all_tasks.extend(eligible_flights)
        all_tasks.extend(crew_ground_duties)

        # 按开始时间排序
        all_tasks.sort(key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))

        # 贪心选择任务，确保满足约束
        selected_tasks = []

        for task in all_tasks:
            # 检查是否可以添加这个任务
            temp_tasks = selected_tasks + [task]

            # 简单的约束检查
            if self._is_valid_task_sequence(temp_tasks):
                selected_tasks.append(task)

                # 限制任务数量，避免过度复杂
                if len(selected_tasks) >= 5:
                    break

        if selected_tasks:
            # 计算roster的成本
            cost = sum(getattr(task, 'cost', 0) for task in selected_tasks)
            roster = Roster(crew.crewId, selected_tasks, cost)
            return roster

        return None

    def _is_valid_task_sequence(self, tasks: List) -> bool:
        """检查任务序列是否有效"""
        if not tasks:
            return True

        # 简单的时间冲突检查
        for i in range(len(tasks) - 1):
            current_task = tasks[i]
            next_task = tasks[i + 1]

            # 获取任务的结束和开始时间
            current_end = getattr(current_task, 'sta', getattr(current_task, 'endTime', None))
            next_start = getattr(next_task, 'std', getattr(next_task, 'startTime', None))

            if current_end and next_start:
                # 检查最小连接时间
                time_gap = next_start - current_end
                if time_gap < timedelta(minutes=30):  # 最小30分钟间隔
                    return False

        return True


class GurobiRepair(RepairOperator):
    """基于Gurobi MILP求解的修复算子"""

    def __init__(self, crews: List[Crew], flights: List[Flight],
                 ground_duties: List[GroundDuty], bus_info: List[BusInfo],
                 crew_leg_match_dict: Dict, layover_stations: Set[str]):
        super().__init__("GurobiRepair")
        self.crews = crews
        self.flights = flights
        self.ground_duties = ground_duties
        self.bus_info = bus_info
        self.crew_leg_match_dict = crew_leg_match_dict
        self.layover_stations = layover_stations

        # 准备全局数据
        self.all_data = {
            'flights': flights,
            'ground_duties': ground_duties,
            'bus_info': bus_info,
            'crews': crews
        }

        # 初始化Gurobi求解器
        self.gurobi_solver = GurobiRepairSolver(self.all_data, layover_stations)

        # 备用贪心修复器（当Gurobi失败时使用）
        self.greedy_repair = GreedyRepair(crews, flights, ground_duties, bus_info,
                                        crew_leg_match_dict, layover_stations)

    def repair(self, solution: ALNSSolution, removed_rosters: List[Roster]) -> ALNSSolution:
        """使用Gurobi MILP求解器修复解决方案 - 正确的ALNS逻辑"""
        new_solution = solution.copy()

        # **正确的ALNS逻辑**：
        # 1. 获取所有需要重新分配的任务（未覆盖 + 被移除的）
        uncovered_tasks = self._get_uncovered_tasks(new_solution)

        removed_tasks = []
        for roster in removed_rosters:
            removed_tasks.extend(roster.duties)

        # 合并所有需要分配的任务
        all_tasks_to_assign = uncovered_tasks.copy()
        for task in removed_tasks:
            task_id = getattr(task, 'id', str(task))
            if not any(getattr(t, 'id', str(t)) == task_id for t in all_tasks_to_assign):
                all_tasks_to_assign.append(task)

        print(f"ALNS修复: 需要重新分配{len(all_tasks_to_assign)}个任务 (未覆盖{len(uncovered_tasks)} + 被移除{len(removed_tasks)})")

        # 2. 获取所有可用的机组（包括被移除rosters的机组）
        available_crews = self._get_available_crews(new_solution, removed_rosters)
        print(f"可用机组: {len(available_crews)}个")

        # **关键**：不进行任何直接恢复，让Gurobi重新优化所有任务分配

        # 3. 使用Gurobi为每个可用机组重新优化任务分配
        successful_repairs = 0
        total_repairs = 0

        for crew in available_crews:
            if not all_tasks_to_assign:
                break

            total_repairs += 1

            # 获取该机组可执行的任务
            candidate_tasks = self._get_candidate_tasks_for_crew(crew, all_tasks_to_assign)

            if not candidate_tasks:
                continue

            print(f"为机组{crew.crewId}求解: {len(candidate_tasks)}个候选任务")

            # 准备修复输入
            repair_input = self._prepare_repair_input(crew, candidate_tasks, new_solution)

            # 调用Gurobi求解器
            result = self.gurobi_solver.solve_local_problem(repair_input, removed_tasks)

            if result.is_feasible and result.new_duties:
                # 创建新的roster
                new_roster = Roster(crew.crewId, result.new_duties, result.cost)
                new_solution.rosters.append(new_roster)

                # 从待分配任务中移除已分配的任务
                all_tasks_to_assign = self._remove_covered_tasks(all_tasks_to_assign, result.new_duties)
                successful_repairs += 1

                print(f"Gurobi成功为机组{crew.crewId}分配{len(result.new_duties)}个任务，成本{result.cost:.2f}")
            else:
                # Gurobi求解失败，尝试简单的贪心方法
                simple_roster = self._create_simple_roster_for_crew(crew, candidate_tasks[:3])
                if simple_roster:
                    new_solution.rosters.append(simple_roster)
                    all_tasks_to_assign = self._remove_covered_tasks(all_tasks_to_assign, simple_roster.duties)
                    print(f"贪心为机组{crew.crewId}分配{len(simple_roster.duties)}个任务")

        # 4. 如果还有未分配的任务，使用贪心修复作为补充
        if all_tasks_to_assign:
            print(f"剩余{len(all_tasks_to_assign)}个未分配任务，使用贪心修复补充")
            # 创建一个临时的removed_rosters来传递给贪心修复
            temp_removed = []
            new_solution = self.greedy_repair.repair(new_solution, temp_removed)

        new_solution._calculate_objective()

        print(f"ALNS修复完成: Gurobi成功{successful_repairs}/{total_repairs}, 最终解{new_solution}")
        return new_solution

    def _get_uncovered_tasks(self, solution: ALNSSolution) -> List[Any]:
        """获取当前解中未覆盖的任务"""
        covered_flight_ids = set()
        covered_ground_duty_ids = set()

        for roster in solution.rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    covered_flight_ids.add(duty.id)
                elif hasattr(duty, 'crewId') and hasattr(duty, 'airport'):
                    covered_ground_duty_ids.add(duty.id)

        uncovered_tasks = []
        uncovered_tasks.extend([f for f in self.flights if f.id not in covered_flight_ids])
        uncovered_tasks.extend([gd for gd in self.ground_duties if gd.id not in covered_ground_duty_ids])

        return uncovered_tasks

    def _get_available_crews(self, solution: ALNSSolution, removed_rosters: List[Roster]) -> List[Crew]:
        """获取可用的机组"""
        assigned_crew_ids = {roster.crew_id for roster in solution.rosters}
        available_crews = [crew for crew in self.crews if crew.crewId not in assigned_crew_ids]

        # 优先考虑被移除roster的机组
        removed_crew_ids = {roster.crew_id for roster in removed_rosters}
        priority_crews = [crew for crew in available_crews if crew.crewId in removed_crew_ids]
        other_crews = [crew for crew in available_crews if crew.crewId not in removed_crew_ids]

        return priority_crews + other_crews

    def _get_candidate_tasks_for_crew(self, crew: Crew, uncovered_tasks: List[Any]) -> List[Any]:
        """获取机组可执行的候选任务"""
        candidate_tasks = []

        # 获取该机组可执行的航班
        eligible_flight_ids = self.crew_leg_match_dict.get(crew.crewId, [])
        for task in uncovered_tasks:
            if isinstance(task, Flight) and task.id in eligible_flight_ids:
                candidate_tasks.append(task)
            elif hasattr(task, 'crewId') and task.crewId == crew.crewId:
                candidate_tasks.append(task)

        # 按时间排序
        candidate_tasks.sort(key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))

        # 动态调整候选任务数量：根据任务类型和重要性
        flight_tasks = [t for t in candidate_tasks if isinstance(t, Flight)]
        ground_tasks = [t for t in candidate_tasks if not isinstance(t, Flight)]

        # 优先保留航班任务，限制地面任务
        max_flights = min(30, len(flight_tasks))  # 最多30个航班
        max_ground = min(10, len(ground_tasks))   # 最多10个地面任务

        selected_tasks = flight_tasks[:max_flights] + ground_tasks[:max_ground]
        selected_tasks.sort(key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))

        return selected_tasks

    def _prepare_repair_input(self, crew: Crew, candidate_tasks: List[Any],
                             solution: ALNSSolution) -> RepairInput:
        """准备Gurobi求解器的输入"""
        # 确定时间窗口
        if candidate_tasks:
            min_time = min(getattr(task, 'std', getattr(task, 'startTime', datetime.max))
                          for task in candidate_tasks)
            max_time = max(getattr(task, 'sta', getattr(task, 'endTime', datetime.min))
                          for task in candidate_tasks)
            time_window = (min_time - timedelta(hours=1), max_time + timedelta(hours=1))
        else:
            now = datetime.now()
            time_window = (now, now + timedelta(days=1))

        # 初始状态
        initial_state = {
            'location': crew.stayStation,
            'time': time_window[0]
        }

        return RepairInput(
            crew=crew,
            candidate_tasks=candidate_tasks,
            time_window=time_window,
            initial_state=initial_state,
            current_solution_rosters=solution.rosters
        )

    def _remove_covered_tasks(self, uncovered_tasks: List[Any], new_duties: List[Any]) -> List[Any]:
        """从未覆盖任务列表中移除已被覆盖的任务"""
        covered_ids = {getattr(duty, 'id', str(duty)) for duty in new_duties}
        return [task for task in uncovered_tasks if getattr(task, 'id', str(task)) not in covered_ids]

    def _create_simple_roster_for_crew(self, crew: Crew, candidate_tasks: List[Any]) -> Optional[Roster]:
        """为机组创建简单的roster（备用方法）"""
        if not candidate_tasks:
            return None

        # 选择第一个可行的任务
        selected_task = candidate_tasks[0]
        cost = getattr(selected_task, 'cost', 0)

        return Roster(crew.crewId, [selected_task], cost)


class RandomRepair(RepairOperator):
    """随机修复算子"""

    def __init__(self, crews: List[Crew], flights: List[Flight],
                 ground_duties: List[GroundDuty], bus_info: List[BusInfo],
                 crew_leg_match_dict: Dict, layover_stations: Set[str]):
        super().__init__("RandomRepair")
        self.greedy_repair = GreedyRepair(crews, flights, ground_duties, bus_info,
                                        crew_leg_match_dict, layover_stations)

    def repair(self, solution: ALNSSolution, removed_rosters: List[Roster]) -> ALNSSolution:
        """使用随机策略修复解决方案"""
        # 简单实现：随机选择一部分被移除的roster重新加入
        new_solution = solution.copy()

        if removed_rosters:
            # 随机选择一部分被移除的roster
            num_to_restore = random.randint(1, max(1, len(removed_rosters) // 2))
            rosters_to_restore = random.sample(removed_rosters, num_to_restore)

            for roster in rosters_to_restore:
                # 检查是否与现有roster冲突
                if not self._conflicts_with_existing(roster, new_solution.rosters):
                    new_solution.rosters.append(copy.deepcopy(roster))

        # 使用贪心修复填补剩余空缺
        new_solution = self.greedy_repair.repair(new_solution, [])

        return new_solution

    def _conflicts_with_existing(self, new_roster: Roster, existing_rosters: List[Roster]) -> bool:
        """检查新roster是否与现有roster冲突"""
        # 检查机组冲突
        for existing_roster in existing_rosters:
            if existing_roster.crew_id == new_roster.crew_id:
                return True

        # 检查任务冲突
        new_tasks = {getattr(duty, 'id', str(duty)) for duty in new_roster.duties}
        for existing_roster in existing_rosters:
            existing_tasks = {getattr(duty, 'id', str(duty)) for duty in existing_roster.duties}
            if new_tasks & existing_tasks:  # 有交集
                return True

        return False


class AdaptiveWeightManager:
    """自适应权重管理器"""

    def __init__(self, operators: List, reaction_factor: float = 0.1):
        self.operators = operators
        self.reaction_factor = reaction_factor
        self.segment_size = 100  # 每100次迭代更新一次权重
        self.iteration_count = 0

        # 初始化权重
        for op in self.operators:
            op.weight = 1.0

    def select_operator(self) -> Any:
        """基于权重选择算子"""
        if not self.operators:
            return None

        # 计算权重总和
        total_weight = sum(op.weight for op in self.operators)
        if total_weight <= 0:
            # 如果所有权重都为0，重置为均等权重
            for op in self.operators:
                op.weight = 1.0
            total_weight = len(self.operators)

        # 轮盘赌选择
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0

        for op in self.operators:
            cumulative_weight += op.weight
            if rand_val <= cumulative_weight:
                return op

        # 如果没有选中（浮点数精度问题），返回最后一个
        return self.operators[-1]

    def update_weights(self):
        """更新算子权重"""
        self.iteration_count += 1

        if self.iteration_count % self.segment_size == 0:
            for op in self.operators:
                if op.usage_count > 0:
                    success_rate = op.success_count / op.usage_count
                    # 基于成功率调整权重
                    op.weight = (1 - self.reaction_factor) * op.weight + self.reaction_factor * success_rate

                    # 重置计数器
                    op.usage_count = 0
                    op.success_count = 0
                else:
                    # 如果没有使用过，保持当前权重
                    pass

                # 确保权重不会太小
                op.weight = max(op.weight, 0.1)


class ALNSAlgorithm:
    """ALNS主算法类"""

    def __init__(self, flights: List[Flight], crews: List[Crew],
                 ground_duties: List[GroundDuty], bus_info: List[BusInfo],
                 crew_leg_match_dict: Dict, layover_stations: Set[str]):
        self.flights = flights
        self.crews = crews
        self.ground_duties = ground_duties
        self.bus_info = bus_info
        self.crew_leg_match_dict = crew_leg_match_dict
        self.layover_stations = layover_stations

        # 初始化算子 - 只使用WorstRosterDestroy和GurobiRepair
        self.destroy_operators = [
            WorstRosterDestroy()
        ]

        self.repair_operators = [
            GurobiRepair(crews, flights, ground_duties, bus_info,
                        crew_leg_match_dict, layover_stations)
        ]

        # 权重管理器
        self.destroy_weight_manager = AdaptiveWeightManager(self.destroy_operators)
        self.repair_weight_manager = AdaptiveWeightManager(self.repair_operators)

        # 算法参数
        self.max_iterations = float('inf')  # 无限迭代
        self.time_limit = 3600  # 恢复1小时运行
        self.destroy_size_min = 2  # 增加最小破坏大小
        self.destroy_size_max = min(10, max(5, len(crews) // 4))  # 动态调整最大破坏大小

        # 模拟退火参数 - 调整为更慢的冷却
        self.initial_temperature = 5000.0  # 增加初始温度
        self.cooling_rate = 0.9995  # 更慢的冷却速度
        self.min_temperature = 0.1  # 降低最小温度

        # 统计信息
        self.iteration_count = 0
        self.best_solution = None
        self.current_solution = None
        self.temperature = self.initial_temperature

        # 多样化策略
        self.last_improvement_iteration = 0
        self.stagnation_limit = 100  # 100次迭代无改进后增加多样化
        self.diversification_factor = 1.0

        # 验证器
        self.coverage_validator = CoverageValidator(min_coverage_rate=0.8)

    def solve(self, initial_solution: ALNSSolution) -> ALNSSolution:
        """执行ALNS算法"""
        print("开始ALNS算法求解...")
        start_time = time.time()

        # 初始化解
        self.current_solution = initial_solution.copy()
        self.best_solution = initial_solution.copy()
        self.temperature = self.initial_temperature

        print(f"初始解: {self.current_solution}")

        # 主循环 - 只受时间限制约束
        iteration = 0
        while True:
            self.iteration_count = iteration

            # 检查时间限制
            if time.time() - start_time > self.time_limit:
                print(f"达到时间限制({self.time_limit}秒)，算法终止")
                break

            # 选择破坏和修复算子
            destroy_op = self.destroy_weight_manager.select_operator()
            repair_op = self.repair_weight_manager.select_operator()

            # 确定破坏大小（考虑多样化因子）
            base_destroy_size = random.randint(self.destroy_size_min,
                                             min(self.destroy_size_max, len(self.current_solution.rosters)))
            destroy_size = min(len(self.current_solution.rosters) - 1,
                             int(base_destroy_size * self.diversification_factor))

            try:
                # 破坏
                destroyed_solution, removed_elements = destroy_op.destroy(self.current_solution, destroy_size)

                # 修复
                new_solution = repair_op.repair(destroyed_solution, removed_elements)

                # 评估新解
                accept_solution = self._should_accept_solution(new_solution)

                # 更新算子权重
                improved = new_solution.is_better_than(self.current_solution)
                destroy_op.update_weight(improved)
                repair_op.update_weight(improved)

                # 接受或拒绝新解
                if accept_solution:
                    self.current_solution = new_solution

                    # 更新最优解
                    if new_solution.is_better_than(self.best_solution):
                        self.best_solution = new_solution.copy()
                        self.last_improvement_iteration = iteration
                        print(f"迭代 {iteration}: 找到更好解 {self.best_solution}")

                # 多样化策略：如果长时间无改进，增加破坏大小和温度
                if iteration - self.last_improvement_iteration > self.stagnation_limit:
                    self.diversification_factor = min(2.0, self.diversification_factor * 1.1)
                    # 重置温度以增加接受概率
                    self.temperature = max(self.temperature, self.initial_temperature * 0.1)
                    print(f"迭代 {iteration}: 启动多样化策略，因子={self.diversification_factor:.2f}")
                else:
                    self.diversification_factor = max(1.0, self.diversification_factor * 0.99)

                # 更新温度
                self.temperature = max(self.min_temperature,
                                     self.temperature * self.cooling_rate)

                # 定期输出进度
                if iteration % 100 == 0:
                    print(f"迭代 {iteration}: 当前解={self.current_solution.objective_value:.2f}, "
                          f"最优解={self.best_solution.objective_value:.2f}, "
                          f"温度={self.temperature:.2f}")

                    # 输出算子使用情况
                    self._print_operator_stats()

                # 更新权重
                self.destroy_weight_manager.update_weights()
                self.repair_weight_manager.update_weights()

            except Exception as e:
                print(f"迭代 {iteration} 出错: {e}")
                continue

            # 增加迭代计数器
            iteration += 1

        print(f"ALNS算法完成，总迭代次数: {self.iteration_count + 1}")
        print(f"最优解: {self.best_solution}")

        return self.best_solution

    def _should_accept_solution(self, new_solution: ALNSSolution) -> bool:
        """判断是否接受新解（模拟退火准则）"""
        if new_solution.is_better_than(self.current_solution):
            return True

        if self.temperature <= self.min_temperature:
            return False

        # 计算接受概率
        delta = new_solution.objective_value - self.current_solution.objective_value
        probability = math.exp(-delta / self.temperature)

        return random.random() < probability

    def _print_operator_stats(self):
        """输出算子使用统计"""
        print("破坏算子统计:")
        for op in self.destroy_operators:
            success_rate = op.success_count / op.usage_count if op.usage_count > 0 else 0
            print(f"  {op.name}: 权重={op.weight:.3f}, 使用={op.usage_count}, 成功率={success_rate:.3f}")

        print("修复算子统计:")
        for op in self.repair_operators:
            success_rate = op.success_count / op.usage_count if op.usage_count > 0 else 0
            print(f"  {op.name}: 权重={op.weight:.3f}, 使用={op.usage_count}, 成功率={success_rate:.3f}")


def main():
    """ALNS主函数"""
    print("=== ALNS机组排班优化系统 ===")

    # 数据加载
    print("正在加载数据...")
    data_path = UnifiedConfig.DATA_PATH
    all_data = load_all_data(data_path)

    if not all_data:
        print("数据加载失败，程序退出。")
        return

    flights = all_data["flights"]
    crews = all_data["crews"]
    bus_info = all_data["bus_info"]
    ground_duties = all_data["ground_duties"]
    crew_leg_match_list = all_data["crew_leg_matches"]
    layover_stations = all_data["layover_stations"]

    # 预处理机长-航班资质数据
    crew_leg_match_dict = {}
    for match in crew_leg_match_list:
        flight_id, crew_id = match.flightId, match.crewId
        if crew_id not in crew_leg_match_dict:
            crew_leg_match_dict[crew_id] = []
        crew_leg_match_dict[crew_id].append(flight_id)

    print(f"数据加载完成: 航班{len(flights)}个, 机组{len(crews)}个, 地面任务{len(ground_duties)}个")

    # 生成初始解
    print("正在生成初始解...")
    initial_rosters = generate_initial_rosters_with_heuristic(
        flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
    )

    if not initial_rosters:
        print("初始解生成失败，程序退出。")
        return

    # 创建ALNS解决方案对象
    initial_solution = ALNSSolution(initial_rosters, flights, ground_duties, crews)
    print(f"初始解生成完成: {initial_solution}")

    # 验证初始解
    coverage_validator = CoverageValidator(min_coverage_rate=0.8)
    coverage_result = coverage_validator.validate_coverage(flights, initial_rosters)
    print(f"初始解航班覆盖率: {coverage_result['coverage_rate']:.2%}")

    # 创建ALNS算法实例
    alns = ALNSAlgorithm(flights, crews, ground_duties, bus_info,
                        crew_leg_match_dict, layover_stations)

    # 执行ALNS算法
    best_solution = alns.solve(initial_solution)

    # 验证最终解
    final_coverage_result = coverage_validator.validate_coverage(flights, best_solution.rosters)
    print(f"\n=== 最终结果 ===")
    print(f"最优解: {best_solution}")
    print(f"航班覆盖率: {final_coverage_result['coverage_rate']:.2%}")
    print(f"地面任务覆盖率: {best_solution.ground_duty_coverage_rate:.2%}")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"output/alns_result_{timestamp}.csv"

    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)

    # 使用简化的结果写入函数
    write_alns_results_to_csv(best_solution.rosters, output_file)
    print(f"结果已保存到: {output_file}")


def write_alns_results_to_csv(rosters: List[Roster], output_file: str):
    """简化的结果写入函数，专门用于ALNS"""
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['crewId', 'dutyId', 'dutyType', 'startTime', 'endTime', 'airport'])

        # 写入每个roster的任务
        for roster in rosters:
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    writer.writerow([
                        roster.crew_id,
                        duty.id,
                        'Flight',
                        duty.std.strftime('%Y-%m-%d %H:%M:%S'),
                        duty.sta.strftime('%Y-%m-%d %H:%M:%S'),
                        f"{duty.depaAirport}-{duty.arriAirport}"
                    ])
                elif hasattr(duty, 'crewId') and hasattr(duty, 'airport'):
                    # 地面任务
                    writer.writerow([
                        roster.crew_id,
                        duty.id,
                        'GroundDuty',
                        duty.startTime.strftime('%Y-%m-%d %H:%M:%S'),
                        duty.endTime.strftime('%Y-%m-%d %H:%M:%S'),
                        duty.airport
                    ])
                elif hasattr(duty, 'depaAirport') and hasattr(duty, 'arriAirport'):
                    # 大巴任务
                    writer.writerow([
                        roster.crew_id,
                        getattr(duty, 'id', 'BUS'),
                        'Bus',
                        duty.startTime.strftime('%Y-%m-%d %H:%M:%S'),
                        duty.endTime.strftime('%Y-%m-%d %H:%M:%S'),
                        f"{duty.depaAirport}-{duty.arriAirport}"
                    ])


if __name__ == "__main__":
    main()