#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化测试：验证基本的破坏-修复功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_loader import load_all_data
    from unified_config import UnifiedConfig
    print("✅ 模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    exit(1)

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    try:
        # 1. 测试数据加载
        print("1. 测试数据加载...")
        data_path = UnifiedConfig.DATA_PATH
        all_data = load_all_data(data_path)
        
        if not all_data:
            print("❌ 数据加载失败")
            return False
        
        print(f"✅ 数据加载成功: 航班{len(all_data['flights'])}个")
        
        # 2. 测试ALNS类导入
        print("2. 测试ALNS类导入...")
        from main import ALNSSolution, WorstRosterDestroy, GurobiRepair
        print("✅ ALNS类导入成功")
        
        # 3. 测试基本对象创建
        print("3. 测试基本对象创建...")
        destroy_op = WorstRosterDestroy()
        print(f"✅ 破坏算子创建成功: {destroy_op.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始简化测试...")
    
    success = test_basic_functionality()
    
    if success:
        print("✅ 基本功能测试通过")
    else:
        print("❌ 基本功能测试失败")
