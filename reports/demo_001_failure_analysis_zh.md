# demo_001 失敗原因分析

## 摘要

這次 `demo_001` 並不是執行失敗，也不是 final score 算錯，而是一次「沒有成功演化」的 run。

核心現象是：baseline genome 在這個 toy scenario 上已經滿足所有可見的格式與需求檢查，因此 evaluator 很快進入飽和狀態。後續 mutation 雖然有產生候選 genome，但沒有任何候選能達到 promotion 門檻，最後系統只好 freeze 原本的最佳 genome。因為最佳 genome 仍然是 baseline genome，所以最後顯示：

- Baseline score: `0.4829`
- Final score: `0.4829`
- Promoted mutations: `0`
- Stop reason: `maximum generation hard cap reached`

## 觀察到的現象

### 1. baseline 與 final 相同

`runs/demo_001/baseline_genome.yaml` 和 `runs/demo_001/frozen_genome.yaml` 完全相同。這代表 final freeze 保存的是原始 baseline genome，而不是某個後續變異版本。

final evaluation 重新跑了一次 frozen genome，得到：

- `promotion_score = 0.48288`
- `train_score = 0.48288`
- `validation_score = 0.48288`

四捨五入後就是 CLI 看到的 `0.4829`。

### 2. 每一代分數幾乎沒有變化

lineage 顯示 generation 0 到 generation 6 的主要分數都維持在 `0.4829`。其中 generation 1 之後有些 raw score 是 `0.48294`，比 baseline 的 `0.48288` 高一點點，但 CLI 與報告四捨五入後都會顯示成 `0.4829`。

這個微小差距只有 `0.00006`，遠低於 scenario 設定的 promotion 門檻：

- `min_delta = 0.03`

所以即使某些候選看起來略有改善，也無法被 Guardian promotion。

### 3. 最後一個 candidate 變差後被 rollback

你看到的這段：

```text
score: 0.4829 -> 0.4769
reason: fitness did not improve enough for promotion
```

意思是 generation 6 的候選 mutation 從目前最佳 `0.4829` 降到 `0.4769`，所以被 rollback。這個 `0.4769` 不是 final score，而是「失敗候選」的 score。

該 candidate 的 eval result 顯示：

- `promotion_score = 0.47688`
- `complexity_penalty = 0.12`
- `requirement_coverage = 1.0`
- `format_validity = 1.0`
- `scenario_success = 1.0`

也就是說，它仍然滿足基本輸出需求，但因為 genome 結構變複雜，受到 complexity penalty，分數反而下降。

## 主要失敗原因

### 原因一：toy scenario 太容易，baseline 已經吃滿可見需求

scenario 是 `toy_structured_answer`，任務是「對模糊請求產生結構化、有用的回答」。可見要求只有：

- Must include a short summary
- Must include concrete steps
- Must include final answer

baseline 輸出已經穩定包含 Summary、Concrete Steps、Final Answer，因此所有 train / validation cases 都拿到：

- `requirement_coverage = 1.0`
- `format_validity = 1.0`
- `scenario_success = 1.0`

這讓後續 mutation 很難在可見 evaluator 裡展現明顯進步。

### 原因二：evaluator 對「回答品質」不夠敏感

這次分數主要反映的是表面結構與 trace 訊號，例如：

- 是否覆蓋必要 section
- markdown 格式是否有效
- workflow 是否完成
- 是否有 artifact
- 是否有 self review
- 是否有 quality gate
- 是否使用 generated tool
- cost / complexity penalty

但它沒有真正細緻地判斷回答是否更具體、更貼近使用者情境、或比 baseline 更有洞察。結果就是：只要輸出有固定段落，分數就很快飽和。

### 原因三：promotion 門檻相對於 score granularity 太高

目前 `min_delta = 0.03`。但這次可觀察到的正向變化大約只有：

- `0.48288 -> 0.48294`
- delta 約 `0.00006`

這表示 promotion 門檻大約是實際微小改善的 500 倍。對這個 toy task 來說，除非 mutation 觸發新的 evaluator reward，例如 quality gate、self review 或 tool usage，否則幾乎不可能被採納。

### 原因四：前幾代 mutation 沒有對準真正瓶頸

generation 0 到 generation 2 的 mutation 都集中在 retry policy，例如：

- 啟用 `revise_on_failure`
- 增加 retry 機制

但 eval result 顯示所有 cases 都沒有 failures。既然 baseline 已經一次成功，retry policy 就不會被觸發，也不會改善 evaluator 能看到的任何訊號。

### 原因五：後期 whole-genome replacement 增加複雜度，但沒有帶來可見收益

generation 3 之後開始使用 zero-order / hyper 類型的 whole-genome replacement。最後一個候選 genome 加入了第二個角色 `StrategyAuditor`，也啟用了 self-evaluation 設定。

問題是這些變化沒有轉化成 evaluator 有獎勵的 trace 或結果：

- `self_review_usage` 仍是 `0.0`
- `quality_gate_usage` 仍是 `0.0`
- `generated_tool_usage` 仍是 `0.0`
- `artifact_presence` 仍是 `0.0`

但 complexity penalty 變成 `0.12`，因此總分從 `0.4829` 掉到 `0.4769`。

### 原因六：搜尋進入 plateau，最後被 generation hard cap 停止

lineage 顯示：

- generation 4 開始偵測到 validation score plateau
- generation 5 plateau persisted
- generation 6 達到 max generation hard cap

最後 decision 是：

```text
action: stop_max_generations
stop: True
reason: maximum generation hard cap reached
plateau detected: True
```

這代表系統不是因為找到滿意改善而停止，而是因為連續沒有有效進步，最後跑到最大世代數限制。

## 根因整理

### 一級根因

- 任務太容易，baseline 已經滿足 evaluator 的主要可見條件。
- evaluator 對「更好的回答品質」缺少足夠解析度。
- promotion 門檻 `0.03` 對這個分數尺度太高。

### 二級根因

- mutation 策略優先改 retry policy，但當前 failure count 是 0，retry 改動沒有作用。
- whole-genome mutation 增加角色與流程，卻沒有實際產生 self-review / quality-gate trace。
- complexity penalty 比新增結構帶來的 reward 更早生效。
- demo 使用的 case 數太少：2 個 train cases、1 個 validation case，容易讓 baseline 用通用模板拿到穩定高分。

## 可放進報告的結論

這次 demo run 的失敗不是「模型無法回答問題」，而是「演化系統沒有取得可被 promotion 規則承認的進步」。baseline 已經用簡單 Founder genome 滿足 toy scenario 的表層要求，導致 evaluator 飽和。後續 mutation 不是沒有影響，就是增加了複雜度但沒有產生可獎勵的 trace。由於 `min_delta = 0.03` 遠高於實際 score 微幅變化，Guardian 正確地拒絕所有候選，最後 freeze 原本 baseline genome。

## 建議改進方向

### 1. 降低 toy scenario 的 promotion 門檻

可以先試：

- `min_delta = 0.001`
- 或更敏感一點：`min_delta = 0.0005`

這能讓小型 task 上的細微改善有機會被 promotion。

### 2. 增加更難的 train / validation cases

目前 prompt 都很泛用，通用 checklist 很容易過關。可以加入：

- 有限制條件的請求，例如時間、預算、風險偏好
- 需要取捨的請求，例如「我只有 15 分鐘」
- 容易被泛泛回答騙過的 vague prompt
- hidden cases，用來懲罰模板化回答

### 3. 讓 evaluator 直接評估 usefulness / specificity

目前 evaluator 對 Summary / Steps / Final Answer 很敏感，但對回答是否真的更好不夠敏感。可以加入：

- input-specific usefulness score
- specificity score
- actionability score
- anti-boilerplate penalty

### 4. 讓 mutation 對準可見 bottleneck

與其先改 retry policy，這類 scenario 更適合先嘗試：

- 增加 requirement review step
- 增加 final answer revision step
- 增加 lightweight quality gate
- 讓 self-evaluation 真正產生 trace，而不只是 genome 裡打開設定

### 5. 避免重用同一個 run id 做比較

如果後續要寫報告或比較實驗，建議每次用新的 run id。這樣 lineage、convergence log、report 都比較乾淨，也比較容易判斷每次修改的效果。

## 下一次實驗建議

建議下一輪使用新的 run id，例如 `demo_002`，並做以下調整：

1. 將 `min_delta` 降到 `0.001`。
2. 增加 3 到 5 個更難的 validation cases。
3. 加入 usefulness / specificity 類型的 scoring signal。
4. 優先測試 review / quality gate mutation，而不是 retry policy mutation。
5. 比較 baseline genome、第一個 promoted genome、final frozen genome 的差異。

如果這樣仍然沒有 promotion，問題就更可能在 mutation planner 無法產生 evaluator 可識別的改善；如果開始出現 promotion，則代表這次失敗主要是 scenario 與 scoring signal 太容易飽和。
