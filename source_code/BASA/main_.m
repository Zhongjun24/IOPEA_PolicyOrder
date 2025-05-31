clear; clc;

% 1) 要测试的 T 列表
Ts = [100000];

% 2) 每个 T 重复次数
nRuns = 20;

% 3) 预分配结果矩阵
results = zeros(length(Ts), nRuns);

% 4) 循环调用
for iT = 1:length(Ts)
    T = Ts(iT);
    fprintf('Running T = %d ...\n', T);
    for r = 1:nRuns
        % 调用 main3，获取 totalcost
        totalcost = main3(T);
        results(iT, r) = totalcost/T;
    end
end
writematrix(results, 'results_with_header.csv')