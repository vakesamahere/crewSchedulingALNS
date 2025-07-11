# file: gurobi_repair_solver.py

import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from data_models import Crew, Flight, Roster, GroundDuty, BusInfo
from unified_config import UnifiedConfig
from constraint_checker import UnifiedConstraintChecker

class RepairInput:
    """封装传递给 Gurobi 求解器的所有输入信息"""
    def __init__(self, crew: Crew, candidate_tasks: List[Any], time_window: tuple,
                 initial_state: dict, current_solution_rosters: List[Roster] = None):
        self.crew = crew
        self.candidate_tasks = candidate_tasks  # 候选任务列表（航班、地面任务等）
        self.time_window = time_window  # (start_time, end_time)
        self.initial_state = initial_state  # {'location': 'PEK', 'time': datetime(...)}
        self.current_solution_rosters = current_solution_rosters or []  # 当前解中的其他rosters

class RepairResult:
    """封装从 Gurobi 求解器返回的结果"""
    def __init__(self, is_feasible: bool, new_duties: List[Any], cost: float,
                 solve_time: float = 0.0, gap: float = 0.0):
        self.is_feasible = is_feasible
        self.new_duties = new_duties
        self.cost = cost
        self.solve_time = solve_time
        self.gap = gap

class GurobiRepairSolver:
    """
    一个专门使用 Gurobi 解决局部修复问题的求解器。
    基于MILP模型为单个机组重新分配任务，满足所有约束条件。
    """
    def __init__(self, all_data: Dict[str, Any], layover_stations: set):
        """
        初始化求解器。

        Args:
            all_data (Dict): 全局数据，用于快速查找任务信息。
            layover_stations (set): 过站机场集合
        """
        self.all_data = all_data
        self.layover_stations = layover_stations

        # 预处理数据，方便快速查找
        self.flights_dict = {f.id: f for f in all_data.get('flights', [])}
        self.ground_duties_dict = {gd.id: gd for gd in all_data.get('ground_duties', [])}
        self.bus_info_dict = {bus.id: bus for bus in all_data.get('bus_info', [])}

        # 初始化约束检查器
        self.constraint_checker = UnifiedConstraintChecker(layover_stations)

        # 时间相关常量
        self.BIG_M = 10000  # 大M常数
        self.TIME_PRECISION = 60  # 时间精度（分钟）

    def solve_local_problem(self, repair_input: RepairInput, removed_tasks: List[Any] = None) -> RepairResult:
        """
        接收一个局部问题，构建并求解 MILP 模型。

        Args:
            repair_input (RepairInput): 描述局部问题的输入对象。

        Returns:
            RepairResult: 包含最优局部排班方案的结果对象。
        """
        try:
            # 1. 创建 Gurobi 模型
            model = gp.Model(f"repair_{repair_input.crew.crewId}")
            model.setParam('OutputFlag', 0)  # 关闭输出
            model.setParam('TimeLimit', 60)  # 增加到60秒时间限制
            model.setParam('MIPGap', 0.05)   # 放宽到5% gap tolerance，提高求解速度
            model.setParam('Heuristics', 0.5)  # 增加启发式搜索
            model.setParam('Cuts', 2)  # 增强切割平面

            # 2. 预处理候选任务
            candidate_tasks = self._preprocess_candidate_tasks(repair_input, removed_tasks or [])
            if not candidate_tasks:
                return RepairResult(is_feasible=False, new_duties=[], cost=float('inf'))

            # 3. 定义决策变量
            task_vars, sequence_vars, time_vars = self._define_variables(model, candidate_tasks, repair_input)

            # 4. 添加约束
            self._add_constraints(model, candidate_tasks, repair_input, task_vars, sequence_vars, time_vars)

            # 5. 设置目标函数
            self._set_objective(model, candidate_tasks, repair_input, task_vars, sequence_vars)

            # 6. 求解模型
            model.optimize()

            # 7. 解析结果
            if model.Status == GRB.OPTIMAL or model.Status == GRB.FEASIBLE:
                new_duties = self._parse_solution(model, candidate_tasks, task_vars, sequence_vars)
                solve_time = model.Runtime
                gap = model.MIPGap if hasattr(model, 'MIPGap') else 0.0
                return RepairResult(
                    is_feasible=True,
                    new_duties=new_duties,
                    cost=model.ObjVal,
                    solve_time=solve_time,
                    gap=gap
                )
            else:
                return RepairResult(is_feasible=False, new_duties=[], cost=float('inf'))

        except gp.GurobiError as e:
            print(f"Gurobi Error: {e.errno} in model for crew {repair_input.crew.crewId}: {e.message}")
            return RepairResult(is_feasible=False, new_duties=[], cost=float('inf'))
        except Exception as e:
            print(f"An unexpected error occurred in Gurobi solver: {e}")
            return RepairResult(is_feasible=False, new_duties=[], cost=float('inf'))

    def _preprocess_candidate_tasks(self, repair_input: RepairInput, removed_tasks: List[Any]) -> List[Dict]:
        """
        预处理候选任务，添加必要的属性和过滤不可行的任务
        """
        processed_tasks = []
        crew = repair_input.crew
        time_window = repair_input.time_window

        # 创建被移除任务的ID集合
        removed_task_ids = {getattr(task, 'id', str(task)) for task in removed_tasks}

        for i, task in enumerate(repair_input.candidate_tasks):
            # 检查任务是否在时间窗口内
            task_start = getattr(task, 'std', getattr(task, 'startTime', None))
            task_end = getattr(task, 'sta', getattr(task, 'endTime', None))

            if not task_start or not task_end:
                continue

            # 检查时间窗口
            if task_start < time_window[0] or task_end > time_window[1]:
                continue

            # 检查是否是被移除的任务
            task_id = getattr(task, 'id', f'task_{i}')
            is_removed_task = task_id in removed_task_ids

            # 创建任务字典
            task_dict = {
                'index': i,
                'original_task': task,
                'id': task_id,
                'start_time': task_start,
                'end_time': task_end,
                'duration': int((task_end - task_start).total_seconds() / 60),  # 分钟
                'type': self._get_task_type(task),
                'location_start': self._get_task_start_location(task),
                'location_end': self._get_task_end_location(task),
                'cost': self._calculate_task_cost(task, crew, is_removed_task),
                'is_flight': isinstance(task, Flight),
                'is_ground_duty': isinstance(task, GroundDuty),
                'is_bus': isinstance(task, BusInfo),
                'is_removed_task': is_removed_task  # 标记是否为被移除的任务
            }

            # 添加飞行特定属性
            if isinstance(task, Flight):
                task_dict.update({
                    'flight_time': getattr(task, 'flyTime', 0),  # 分钟
                    'aircraft_no': getattr(task, 'aircraftNo', None)
                })

            processed_tasks.append(task_dict)

        return processed_tasks

    def _get_task_type(self, task) -> str:
        """获取任务类型"""
        if isinstance(task, Flight):
            return 'flight'
        elif isinstance(task, GroundDuty):
            return 'ground_duty'
        elif isinstance(task, BusInfo):
            return 'bus'
        else:
            return 'unknown'

    def _get_task_start_location(self, task) -> str:
        """获取任务起始位置"""
        if hasattr(task, 'depaAirport'):
            return task.depaAirport
        elif hasattr(task, 'airport'):
            return task.airport
        else:
            return 'UNKNOWN'

    def _get_task_end_location(self, task) -> str:
        """获取任务结束位置"""
        if hasattr(task, 'arriAirport'):
            return task.arriAirport
        elif hasattr(task, 'airport'):
            return task.airport
        else:
            return 'UNKNOWN'

    def _calculate_task_cost(self, task, crew: Crew, is_removed_task: bool = False) -> float:
        """计算任务成本"""
        cost = 0.0

        if isinstance(task, Flight):
            # 飞行任务：飞行时间奖励（负成本）
            flight_time_hours = getattr(task, 'flyTime', 0) / 60.0
            cost -= flight_time_hours * UnifiedConfig.FLIGHT_TIME_REWARD

            # **关键修复**：如果是被移除的任务，给予额外奖励
            if is_removed_task:
                cost -= 100.0  # 额外奖励，鼓励选择被移除的任务

        elif isinstance(task, GroundDuty):
            # 地面任务：基础成本
            cost += 1.0

            # 被移除的地面任务也给予奖励
            if is_removed_task:
                cost -= 50.0

        elif isinstance(task, BusInfo):
            # 大巴任务：置位惩罚
            cost += UnifiedConfig.POSITIONING_PENALTY

            # 被移除的大巴任务减少惩罚
            if is_removed_task:
                cost -= 25.0

        return cost

    def _define_variables(self, model: gp.Model, candidate_tasks: List[Dict],
                         repair_input: RepairInput) -> Tuple[Dict, Dict, Dict]:
        """
        定义MILP模型的决策变量

        Returns:
            Tuple[task_vars, sequence_vars, time_vars]
        """
        n_tasks = len(candidate_tasks)

        # 1. 任务选择变量: x[i] = 1 if task i is selected
        task_vars = {}
        for i in range(n_tasks):
            task_vars[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

        # 2. 任务顺序变量: y[i,j] = 1 if task i is immediately before task j
        sequence_vars = {}
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i != j:
                    sequence_vars[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

        # 3. 时间变量: t[i] = start time of task i (in minutes from reference time)
        time_vars = {}
        reference_time = repair_input.time_window[0]
        max_time = int((repair_input.time_window[1] - reference_time).total_seconds() / 60)

        for i in range(n_tasks):
            # 任务的实际开始时间（相对于参考时间的分钟数）
            task_start_minutes = int((candidate_tasks[i]['start_time'] - reference_time).total_seconds() / 60)
            time_vars[i] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=task_start_minutes,
                ub=task_start_minutes,  # 固定为任务的实际开始时间
                name=f"t_{i}"
            )

        # 4. 辅助变量：机组位置变量 (用于位置连接约束)
        # location_vars[i] = 机组在任务i开始时的位置编码
        # 这里简化处理，通过约束来确保位置连接正确

        model.update()
        return task_vars, sequence_vars, time_vars

    def _add_constraints(self, model: gp.Model, candidate_tasks: List[Dict],
                        repair_input: RepairInput, task_vars: Dict,
                        sequence_vars: Dict, time_vars: Dict):
        """
        添加MILP模型的约束条件
        """
        n_tasks = len(candidate_tasks)
        crew = repair_input.crew

        # 1. 顺序约束：如果任务i被选中且任务j被选中，则必须有明确的顺序关系
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i != j:
                    # 如果两个任务都被选中，则必须有一个在另一个之前
                    model.addConstr(
                        task_vars[i] + task_vars[j] <= 1 + sequence_vars.get((i, j), 0) + sequence_vars.get((j, i), 0),
                        name=f"order_{i}_{j}"
                    )

        # 2. 时间冲突约束：重叠的任务不能同时被选中
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                if self._tasks_overlap(candidate_tasks[i], candidate_tasks[j]):
                    model.addConstr(
                        task_vars[i] + task_vars[j] <= 1,
                        name=f"no_overlap_{i}_{j}"
                    )

        # 3. 位置连接约束：连续任务的位置必须匹配
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i != j and (i, j) in sequence_vars:
                    # 如果任务i直接在任务j之前，检查位置连接
                    if not self._can_connect_tasks(candidate_tasks[i], candidate_tasks[j]):
                        model.addConstr(sequence_vars[i, j] == 0, name=f"location_connect_{i}_{j}")

        # 4. 时间间隔约束：连续任务之间必须有足够的间隔时间
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i != j and (i, j) in sequence_vars:
                    min_gap = self._get_min_connection_time(candidate_tasks[i], candidate_tasks[j])
                    # 如果i在j之前，则j的开始时间 >= i的结束时间 + 最小间隔
                    task_i_end_minutes = int((candidate_tasks[i]['end_time'] - repair_input.time_window[0]).total_seconds() / 60)
                    model.addConstr(
                        time_vars[j] >= task_i_end_minutes + min_gap - self.BIG_M * (1 - sequence_vars[i, j]),
                        name=f"time_gap_{i}_{j}"
                    )

        # 5. 值勤时间限制：选中任务的总值勤时间不超过限制
        if candidate_tasks:
            total_duty_time = gp.quicksum(
                task_vars[i] * candidate_tasks[i]['duration'] for i in range(n_tasks)
            )
            model.addConstr(
                total_duty_time <= UnifiedConfig.MAX_DUTY_DAY_HOURS * 60,  # 转换为分钟
                name="max_duty_time"
            )

        # 6. 飞行任务数量限制
        flight_tasks = [i for i in range(n_tasks) if candidate_tasks[i]['is_flight']]
        if flight_tasks:
            model.addConstr(
                gp.quicksum(task_vars[i] for i in flight_tasks) <= UnifiedConfig.MAX_FLIGHTS_IN_DUTY,
                name="max_flights"
            )

        # 7. 总任务数量限制
        model.addConstr(
            gp.quicksum(task_vars[i] for i in range(n_tasks)) <= UnifiedConfig.MAX_TASKS_IN_DUTY,
            name="max_tasks"
        )

        # 8. 飞行时间限制：值勤内总飞行时间不超过限制
        flight_time_expr = gp.quicksum(
            task_vars[i] * candidate_tasks[i].get('flight_time', 0)
            for i in range(n_tasks) if candidate_tasks[i]['is_flight']
        )
        if flight_tasks:
            model.addConstr(
                flight_time_expr <= UnifiedConfig.MAX_FLIGHT_TIME_IN_DUTY_HOURS * 60,  # 转换为分钟
                name="max_flight_time"
            )

    def _tasks_overlap(self, task1: Dict, task2: Dict) -> bool:
        """检查两个任务是否在时间上重叠"""
        return not (task1['end_time'] <= task2['start_time'] or task2['end_time'] <= task1['start_time'])

    def _can_connect_tasks(self, task1: Dict, task2: Dict) -> bool:
        """检查两个任务是否可以连接（位置匹配）"""
        return task1['location_end'] == task2['location_start']

    def _get_min_connection_time(self, task1: Dict, task2: Dict) -> int:
        """获取两个任务之间的最小连接时间（分钟）"""
        if task2['is_flight']:
            if task1['is_flight'] and task1.get('aircraft_no') == task2.get('aircraft_no'):
                return UnifiedConfig.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES
            else:
                return UnifiedConfig.MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS * 60
        elif task2['is_bus']:
            return UnifiedConfig.MIN_CONNECTION_TIME_BUS_HOURS * 60
        else:
            return 30  # 默认30分钟

    def _set_objective(self, model: gp.Model, candidate_tasks: List[Dict],
                      repair_input: RepairInput, task_vars: Dict, sequence_vars: Dict):
        """
        设置MILP模型的目标函数
        目标：最小化总成本（最大化收益）
        """
        n_tasks = len(candidate_tasks)

        # 任务成本：选中任务的成本之和
        task_cost = gp.quicksum(
            task_vars[i] * candidate_tasks[i]['cost'] for i in range(n_tasks)
        )

        # 位置变化惩罚：如果机组需要从初始位置移动到第一个任务
        initial_location = repair_input.initial_state.get('location', repair_input.crew.stayStation)
        positioning_cost = 0

        for i in range(n_tasks):
            if candidate_tasks[i]['location_start'] != initial_location:
                # 如果任务i是第一个被选中的任务且需要置位
                is_first_task = 1 - gp.quicksum(sequence_vars.get((j, i), 0) for j in range(n_tasks) if j != i)
                positioning_cost += task_vars[i] * is_first_task * UnifiedConfig.POSITIONING_PENALTY

        # 总目标函数：最小化成本
        total_cost = task_cost + positioning_cost
        model.setObjective(total_cost, GRB.MINIMIZE)

    def _parse_solution(self, model: gp.Model, candidate_tasks: List[Dict],
                       task_vars: Dict, sequence_vars: Dict) -> List[Any]:
        """
        从已求解的 Gurobi 模型中解析出任务序列
        """
        selected_tasks = []
        n_tasks = len(candidate_tasks)

        # 1. 找出所有被选中的任务
        selected_indices = []
        for i in range(n_tasks):
            if task_vars[i].X > 0.5:  # 二进制变量大于0.5认为被选中
                selected_indices.append(i)

        if not selected_indices:
            return []

        # 2. 根据顺序变量重建任务序列
        if len(selected_indices) == 1:
            # 只有一个任务
            selected_tasks = [candidate_tasks[selected_indices[0]]['original_task']]
        else:
            # 多个任务，需要排序
            task_sequence = self._reconstruct_sequence(selected_indices, sequence_vars, candidate_tasks)
            selected_tasks = [candidate_tasks[i]['original_task'] for i in task_sequence]

        return selected_tasks

    def _reconstruct_sequence(self, selected_indices: List[int], sequence_vars: Dict,
                             candidate_tasks: List[Dict]) -> List[int]:
        """
        根据顺序变量重建任务序列
        """
        if len(selected_indices) <= 1:
            return selected_indices

        # 构建邻接关系
        next_task = {}
        for i in selected_indices:
            for j in selected_indices:
                if i != j and (i, j) in sequence_vars and sequence_vars[i, j].X > 0.5:
                    next_task[i] = j
                    break

        # 找到起始任务（没有前驱的任务）
        has_predecessor = set(next_task.values())
        start_tasks = [i for i in selected_indices if i not in has_predecessor]

        if not start_tasks:
            # 如果没有明确的起始任务，按时间排序
            return sorted(selected_indices, key=lambda i: candidate_tasks[i]['start_time'])

        # 从起始任务开始构建序列
        sequence = []
        current = start_tasks[0]
        visited = set()

        while current is not None and current not in visited:
            sequence.append(current)
            visited.add(current)
            current = next_task.get(current)

        # 添加剩余未访问的任务（按时间排序）
        remaining = [i for i in selected_indices if i not in visited]
        remaining.sort(key=lambda i: candidate_tasks[i]['start_time'])
        sequence.extend(remaining)

        return sequence