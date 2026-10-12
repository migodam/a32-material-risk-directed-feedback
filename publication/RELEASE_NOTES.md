# A32 E0：连续材料恢复的完整负结果

32个已有Maxwell条件、16个已暴露parents；PCA64＋TRAIN Gaussian＋single-anchor calibrated-L的复杂材料MSE相对N3增加39.46%。预登记E0门槛FAIL，E1／三维NOT_RUN。50项接口、数学和QP测试通过；它们不代表成像收益。新增全波状态、训练与locked读取均为零。

`A32_E0_RAW_ARRAYS.zip`包含32份online预测，以及明确隔离的offline重建／诊断数组；解压至仓库根目录。指标、图、源码、配置、原始运行与失败日志保存在Git仓库。完整源数据下载路径见README中的A29 Release。

同固定chart的离线truth-fit仍有明显容量；当前测量驱动先验拟合没有利用它。A17原生材料风险预测与加速没有执行，结论为NOT_ESTABLISHED，不能把此结果解释成A17恒等式失效。
