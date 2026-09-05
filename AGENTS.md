# 工程约定

- 此工程是 CUDA → Metal 迁移的新仓库。迁移阶段见 `docs/MIGRATION.md`，现有证据见 `docs/VALIDATION.md`。
- `../diff-gaussian-rasterization` 是参考源；保留其中用户已有修改。新实现写在本工程。
- 长期架构为 C++ 核心 + Objective-C++ Metal 桥接 + MSL，使用 CMake/CTest。不要恢复 Swift 实现。
- 最终目标是完整复刻固定上游版本的可微渲染功能与 Python API，仅替换 CUDA 后端为 Metal。不能以只做 Forward 或手机 viewer 缩减目标。
- 修改渲染数学、排序、同步或 buffer 布局后运行 CMake 构建与 `ctest --test-dir build --output-on-failure`，用实际 Metal 结果与独立 CPU reference 对照。GPU 不可用时明确记录跳过。
- 修改训练内核或 Tensor 绑定后构建 `_C`，运行 `.venv/bin/python -m pytest tests/python -q`，验证各输入路径的 Forward/Backward、有限差分和优化任务。可用 `.venv/bin/python -m pip install -e . --no-build-isolation --no-deps` 构建安装。
- 不得把 Forward 通过描述为 Backward、CUDA 全量等价或可微训练已经完成。不要通过放宽容差隐藏排序与阈值分支差异。
- 保留上游版权和 `LICENSE.md`。C++/MSL 内部布局不是公开资产 ABI。
- 中文说明数学时采用 Markdown + LaTeX：行内使用 `$...$`，独立公式使用各自单行的 `$$`；公式不放入代码块或反引号。重要公式后解释变量与计算方向，复杂推导逐步展示，并照顾数学初学者。
