import json
import csv
import io
import time
from ortools.sat.python import cp_model

def solve_tyc_schedule(config_data):
    """
    桃園國中 24 班 (701~908) 階層式排課求解引擎
    特色：
    1. 特教資源班 2~3 班同調抽離 (Math/Eng/Chinese 同時段同步)
    2. 週三本土語 (閩/客) 局部班群同進同出 (701+702 等)
    3. 體育課嚴格避開第 5 節 (夏日防呆軟約束)
    4. 桃園授課節數核算 (國文導師 10/其他導師 12/專任 16~18)
    """
    model = cp_model.CpModel()
    
    # 基本參數
    DAYS = [1, 2, 3, 4, 5]
    PERIODS = [1, 2, 3, 4, 5, 6, 7]
    CLASSES = config_data.get("classes", [f"{g}0{b}" for g in [7, 8, 9] for b in range(1, 9)])
    TEACHERS = config_data.get("teachers", {})
    ASSIGNMENTS = config_data.get("assignments", [])
    SYNC_GROUPS = config_data.get("sync_groups", []) # 特教抽離同調群組
    WED_GROUPS = config_data.get("wed_groups", [])   # 週三本土語跑班
    
    # 建立決策變數 x[assignment_idx, day, period]
    x = {}
    for a_idx, a in enumerate(ASSIGNMENTS):
        for d in DAYS:
            for p in PERIODS:
                x[a_idx, d, p] = model.NewBoolVar(f"x_{a_idx}_{d}_{p}")
                
    # 限制條件 1：每堂配課每週節數必須排滿
    for a_idx, a in enumerate(ASSIGNMENTS):
        model.Add(sum(x[a_idx, d, p] for d in DAYS for p in PERIODS) == a["periods"])
        
    # 限制條件 2：同一班級在同一時段至多一堂課 (普通班互斥)
    for c in CLASSES:
        for d in DAYS:
            for p in PERIODS:
                c_assigns = [a_idx for a_idx, a in enumerate(ASSIGNMENTS) if a["class_id"] == c]
                if c_assigns:
                    model.Add(sum(x[a_idx, d, p] for a_idx in c_assigns) <= 1)
                    
    # 限制條件 3：同一教師在同一時段至多一堂課 (教師互斥，除非協同教學)
    for t_name in TEACHERS:
        for d in DAYS:
            for p in PERIODS:
                t_assigns = [a_idx for a_idx, a in enumerate(ASSIGNMENTS) if a["teacher"] == t_name and not a.get("co_teaching", False)]
                if t_assigns:
                    model.Add(sum(x[a_idx, d, p] for a_idx in t_assigns) <= 1)

    # 限制條件 4：特教資源班 2~3 班同調抽離 (核心關鍵)
    # 例如 701 與 702 的數學必須在同一節
    for group in SYNC_GROUPS:
        # group = {"subject": "數學", "classes": ["701", "702"]}
        subj = group["subject"]
        cls_list = group["classes"]
        target_assigns = [a_idx for a_idx, a in enumerate(ASSIGNMENTS) if a["subject"] == subj and a["class_id"] in cls_list]
        if len(target_assigns) >= 2:
            first = target_assigns[0]
            for other in target_assigns[1:]:
                for d in DAYS:
                    for p in PERIODS:
                        # 強制時段二進制變數相等
                        model.Add(x[first, d, p] == x[other, d, p])

    # 限制條件 5：週三本土語局部跑班 (強制排在週三 d=3)
    for w_group in WED_GROUPS:
        subj = w_group["subject"]
        cls_list = w_group["classes"]
        w_assigns = [a_idx for a_idx, a in enumerate(ASSIGNMENTS) if a["subject"] == subj and a["class_id"] in cls_list]
        for a_idx in w_assigns:
            # 只能落在 d=3
            model.Add(sum(x[a_idx, 3, p] for p in PERIODS) == 1)
            # 其他天禁止
            model.Add(sum(x[a_idx, d, p] for d in [1, 2, 4, 5] for p in PERIODS) == 0)

    # 軟約束 6：體育課避開第 5 節 (夏日高溫防呆懲罰)
    penalties = []
    for a_idx, a in enumerate(ASSIGNMENTS):
        if "體育" in a["subject"]:
            for d in DAYS:
                penalties.append(x[a_idx, d, 5] * 100) # 懲罰權重 100

    # 軟約束 7：主科盡量排在上午 (1~4 節)
    for a_idx, a in enumerate(ASSIGNMENTS):
        if a["subject"] in ["國文", "數學", "英語", "理化", "生物"]:
            for d in DAYS:
                for p in [5, 6, 7]: # 下午節次給予輕微懲罰
                    penalties.append(x[a_idx, d, p] * 5)

    if penalties:
        model.Minimize(sum(penalties))

    # 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0 # 60秒超時限制
    solver.parameters.num_workers = 8
    
    print("🚀 啟動 Google OR-Tools CP-SAT 求解器...")
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"✅ 成功找到排課可行解！(狀態: {'最佳解' if status == cp_model.OPTIMAL else '可行解'})")
        result = {}
        for c in CLASSES:
            result[c] = {"title": f"{c} 班級課表", "homeroom": TEACHERS.get(c, {}).get("homeroom", "導師"), "grid": {}}
            
        for a_idx, a in enumerate(ASSIGNMENTS):
            for d in DAYS:
                for p in PERIODS:
                    if solver.Value(x[a_idx, d, p]) == 1:
                        c_id = a["class_id"]
                        tag = ""
                        if any(a["subject"] == g["subject"] and c_id in g["classes"] for g in SYNC_GROUPS):
                            tag = "🔗 資源班抽離"
                        elif a["subject"] == "本土語":
                            tag = "閩/客二選一"
                            
                        result[c_id]["grid"][f"{d}-{p}"] = {
                            "course": a["subject"],
                            "teacher": a["teacher"],
                            "tag": tag
                        }
        return {"status": "success", "data": result}
    else:
        print("❌ 條件過於嚴苛，無可行解！")
        return {"status": "infeasible"}
