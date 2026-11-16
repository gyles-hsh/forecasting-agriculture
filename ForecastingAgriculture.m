% Clear Workspace before Starting
clear; 
clc;
close all;

% Load and Preprocess Data
data = readtable("agriculture_dataset.csv");
season_column = 9;
season_names = lower(string(data{:, season_column}));
month_labels = strings(height(data), 1);

for i = 1:height(data)
    switch season_names(i)
        case 'kharif'
            month_labels(i) = "June–October";
        case 'zaid'
            month_labels(i) = "March–May";
        case 'rabi'
            month_labels(i) = "November–February";
        otherwise
            error("Error: Unknown Season | %d: %s", i, season_names(i));
    end
end

data{:, season_column} = cellstr(month_labels);
data.Farm_ID = [];

target = 'Yield_tons_';
categoricalVars = {'Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season'};
numericVars = {'Farm_Area_acres_', 'Fertilizer_Used_tons_', 'Pesticide_Used_kg_', 'Water_Usage_cubicMeters_'};

X = data(:, setdiff(data.Properties.VariableNames, {target}));
y = data.(target);

% Target Encoding for Categorical Features (with Cross-validation)
cv_targetEncoding = cvpartition(height(X), 'KFold', 5);
X_encoded = X;

for i = 1:numel(categoricalVars)
    var = categoricalVars{i};
    encodedValues = zeros(height(X), 1);

    categoricalColumns = categorical(data{:, var});

    for fold = 1:cv_targetEncoding.NumTestSets
        trainIdx = training(cv_targetEncoding, fold);
        testIdx = test(cv_targetEncoding, fold);

        trainCategories = categoricalColumns(trainIdx);
        testCategories = categoricalColumns(testIdx);
        yTrain = y(trainIdx);

        categoryList = categories(trainCategories);
        categoryMeanMap = containers.Map();

        for j = 1:numel(categoryList)
            c = categoryList{j};
            idx = trainCategories == c;
            categoryMean = mean(yTrain(idx));
            categoryMeanMap(c) = categoryMean;
        end

        testRows = find(testIdx);  
        for j = 1:numel(testRows)
            c = char(testCategories(j));
            if isKey(categoryMeanMap, c)
                encodedValues(testRows(j)) = categoryMeanMap(c);
            else
                encodedValues(testRows(j)) = mean(yTrain);
            end
        end
    end

    X_encoded.(var) = encodedValues;
end

X = X_encoded;

% Engineered Features
X.Fertilizer_Efficiency = zscore(y ./ data.Fertilizer_Used_tons_);
X.Water_Efficiency = zscore(y ./ data.Water_Usage_cubicMeters_);
X.Chemical_Intensity = zscore((data.Fertilizer_Used_tons_ + data.Pesticide_Used_kg_) ./ data.Farm_Area_acres_);

% Group and Score Features
N_category = 2; % Force to Select 2 Categorical Features
N_numerical = 5; % Force to Select 5 Numerical Features

varNames = X.Properties.VariableNames;
groupMap = containers.Map;
groupType = containers.Map;

for i = 1:length(varNames)
    v = varNames{i};
    if ismember(v, categoricalVars)
        group = v; groupType(group) = 'categorical';
    elseif ismember(v, {'Fertilizer_Efficiency', 'Water_Efficiency', 'Chemical_Intensity'})
        group = v; groupType(group) = 'numerical';
    elseif contains(v, 'Fertilizer_Used')
        group = 'Fertilizer_Used_tons_'; groupType(group) = 'numerical';
    elseif contains(v, 'Pesticide_Used')
        group = 'Pesticide_Used_kg_'; groupType(group) = 'numerical';
    elseif contains(v, 'Water_Usage')
        group = 'Water_Usage_cubicMeters_'; groupType(group) = 'numerical';
    elseif contains(v, 'Farm_Area')
        group = 'Farm_Area_acres_'; groupType(group) = 'numerical';
    else
        group = v; groupType(group) = 'numerical';
    end

    if isKey(groupMap, group)
        groupMap(group) = [groupMap(group), i];
    else
        groupMap(group) = i;
    end
end

Xmat_all = table2array(X);
quickModel = fitrensemble(Xmat_all, y, 'Method', 'Bag');
importances = predictorImportance(quickModel);

groupNames = keys(groupMap);
groupScores = zeros(size(groupNames));
for i = 1:length(groupNames)
    idxs = groupMap(groupNames{i});
    groupScores(i) = sum(importances(idxs));
end

categoricalGroups = groupNames(strcmp(values(groupType, groupNames), 'categorical'));
categoricalScores = groupScores(ismember(groupNames, categoricalGroups));
[~, categoryIdx] = sort(categoricalScores, 'descend');
sortedCategoricalGroups = categoricalGroups(categoryIdx);

numericalGroups = groupNames(strcmp(values(groupType, groupNames), 'numerical'));
numericalScores = groupScores(ismember(groupNames, numericalGroups));
[~, numIdx] = sort(numericalScores, 'descend');
sortedNumericalGroups = numericalGroups(numIdx);

selectedCategoricalGroups = sortedCategoricalGroups(1:min(N_category, numel(sortedCategoricalGroups)));
selectedNumericalGroups = sortedNumericalGroups(1:min(N_numerical, numel(sortedNumericalGroups)));
selectedGroups = [selectedCategoricalGroups, selectedNumericalGroups];

selectedIndices = cell2mat(values(groupMap, selectedGroups));
selectedIndices = sort(selectedIndices);
X = X(:, selectedIndices);
Xmat = table2array(X);
selectedFeatures = X.Properties.VariableNames;

% Display Ranked and Selected Features
fprintf('\n=== Ranked Feature Groups (Categorical): ===\n');
for i = 1:length(sortedCategoricalGroups)
    fprintf('%2d. %s\n', i, sortedCategoricalGroups{i});
end

fprintf('\n=== Ranked Feature Groups (Numerical): ===\n');
for i = 1:length(sortedNumericalGroups)
    fprintf('%2d. %s\n', i, sortedNumericalGroups{i});
end

fprintf('\n=== Overall Ranked Feature Groups (Combined): ===\n');
[~, sortedIdxAll] = sort(groupScores, 'descend');
for i = 1:length(sortedIdxAll)
    fprintf('%2d. %s\n', i, groupNames{sortedIdxAll(i)});
end

fprintf('\n=== Selected Features for Model ===\n');
disp(selectedFeatures');

% Normalize for NN and SVM Models
[Xmat_norm, mu, sigma] = zscore(Xmat);

% Hyperparameter Tuning
% Random Forest
tunedRF = fitrensemble(Xmat, y, ...
    'Method', 'Bag', ...
    'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize', 'MaxNumSplits', 'NumVariablesToSample'}, ...
    'HyperparameterOptimizationOptions', struct('KFold', 3, 'MaxObjectiveEvaluations', 25, 'Verbose', 1, 'ShowPlots', false));

% Neural Network
tunedNN = fitrnet(Xmat_norm, y, ...
    'OptimizeHyperparameters', {'LayerSizes','Lambda'}, ...
    'HyperparameterOptimizationOptions', struct('KFold',3,'MaxObjectiveEvaluations',20,'Verbose',1,'ShowPlots',false));

% Support Vector Machine
tunedSVM = fitrsvm(Xmat_norm, y, ...
    'OptimizeHyperparameters', {'KernelFunction','BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct('KFold',3,'MaxObjectiveEvaluations',20,'Verbose',1,'ShowPlots',false));

% Get Best Models
% Random Forest
bestRF = tunedRF.HyperparameterOptimizationResults.XAtMinEstimatedObjective;
rfTemplate = templateTree('MinLeafSize',bestRF.MinLeafSize,'MaxNumSplits',bestRF.MaxNumSplits,'NumPredictorsToSample',bestRF.NumVariablesToSample);
finalRF = fitrensemble(Xmat, y, 'Method','Bag', 'NumLearningCycles',bestRF.NumLearningCycles,'Learners',rfTemplate);

% Neural Network
bestNN = tunedNN.HyperparameterOptimizationResults.XAtMinEstimatedObjective;
bestNNstruct = table2struct(bestNN);
numLayers = bestNNstruct.NumLayers;
layerSizes = zeros(1, numLayers);
for i = 1:numLayers
    fieldName = sprintf('Layer_%d_Size', i);
    layerSizes(i) = bestNNstruct.(fieldName);
end
finalNN = fitrnet(Xmat_norm, y, 'LayerSizes', layerSizes, 'Lambda', bestNNstruct.Lambda, 'Standardize', false);

% Support Vector Machine
bestSVM = tunedSVM.HyperparameterOptimizationResults.XAtMinEstimatedObjective;
if isfield(bestSVM, 'KernelFunction')
    kernelFunc = bestSVM.KernelFunction;
    if iscell(kernelFunc)
        kernelFunc = kernelFunc{1};
    end
    if ~ischar(kernelFunc) && ~isstring(kernelFunc)
        kernelFunc = 'linear';
    end
else
    kernelFunc = 'linear';
end
kernelFunc = char(kernelFunc);

if isfield(bestSVM, 'KernelScale')
    kernelScale = bestSVM.KernelScale;
    if iscell(kernelScale)
        kernelScale = kernelScale{1};
    end
    if isempty(kernelScale) || ~isnumeric(kernelScale) || isnan(kernelScale) || kernelScale <= 0
        kernelScale = 'auto';
    end
else
    kernelScale = 'auto';
end
finalSVM = fitrsvm(Xmat_norm, y, 'KernelFunction', kernelFunc, 'BoxConstraint', bestSVM.BoxConstraint, 'KernelScale', kernelScale, 'Standardize', false);

% Evaluate Models with Repeated K-Fold
models = struct('name', {'Random Forest (RF)', 'Neural Network (NN)', 'Support Vector Machine (SVM)'}, ...
                'model', {finalRF, finalNN, finalSVM});
repeats = 15;
k = 3;
all_R2_all = {};
all_MAE_all = {};
all_RMSE_all = {};

for m = 1:length(models)
    modelName = models(m).name;
    all_R2 = zeros(repeats, 1);
    all_MAE = zeros(repeats, 1);
    all_RMSE = zeros(repeats, 1);

    for r = 1:repeats
        cv = cvpartition(length(y), 'KFold', k);
        yTrue = [];
        yPred = [];

        for fold = 1:k
            trainIdx = training(cv, fold);
            testIdx = test(cv, fold);
            y_train = y(trainIdx);

            switch modelName
                case 'Random Forest (RF)'
                    X_train = Xmat(trainIdx, :);
                    X_test = Xmat(testIdx, :);
                    fitted = fitrensemble(X_train, y_train, ...
                        'Method', 'Bag', ...
                        'NumLearningCycles', bestRF.NumLearningCycles, ...
                        'Learners', rfTemplate);

                case 'Neural Network (NN)'
                    X_train = Xmat_norm(trainIdx, :);
                    X_test = Xmat_norm(testIdx, :);
                    fitted = fitrnet(X_train, y_train, ...
                        'LayerSizes', layerSizes, ...
                        'Lambda', bestNNstruct.Lambda, ...
                        'Standardize', false);

                case 'Support Vector Machine (SVM)'
                    X_train = Xmat_norm(trainIdx, :);
                    X_test = Xmat_norm(testIdx, :);
                    fitted = fitrsvm(X_train, y_train, ...
                        'KernelFunction', kernelFunc, ...
                        'BoxConstraint', bestSVM.BoxConstraint, ...
                        'KernelScale', kernelScale, ...
                        'Standardize', false);
            end

            preds = predict(fitted, X_test);
            yTrue = [yTrue; y(testIdx)];
            yPred = [yPred; preds];
        end

        SSres = sum((yTrue - yPred).^2);
        SStot = sum((yTrue - mean(yTrue)).^2);
        all_R2(r) = 1 - SSres / SStot;
        all_MAE(r) = mean(abs(yTrue - yPred));
        all_RMSE(r) = sqrt(mean((yTrue - yPred).^2));
    end

    all_R2_all{m} = all_R2;
    all_MAE_all{m} = all_MAE;
    all_RMSE_all{m} = all_RMSE;

    fprintf('\n--| %s |--\n', modelName);
    fprintf('R² Mean: %.2f%%\n', mean(all_R2)*100);
    fprintf('MAE Mean: %.3f\n', mean(all_MAE));
    fprintf('RMSE Mean: %.3f\n', mean(all_RMSE));

    % Figure with 3 Boxplots (R², MAE, RMSE) for Current Model
    metricNames = {'R² (%)', 'MAE', 'RMSE'};
    rawMetrics = {all_R2 * 100, all_MAE, all_RMSE};

    figure('Name', sprintf('%s - Evaluation Metrics', modelName), 'NumberTitle', 'off');
    tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    for metricIdx = 1:3
        metricValues = rawMetrics{metricIdx};
        metricLabel = metricNames{metricIdx};

        q1 = quantile(metricValues, 0.25);
        q3 = quantile(metricValues, 0.75);
        minVal = min(metricValues);
        maxVal = max(metricValues);
        medVal = median(metricValues);
        meanVal = mean(metricValues);

        nexttile;
        boxplot(metricValues, 'Labels', {metricLabel});
        title(metricLabel);
        ylabel(metricLabel);
        grid on;

        hold on;
        ylims = ylim;
        ySpacing = (ylims(2) - ylims(1)) * 0.05;

        text(1.1, minVal - ySpacing, sprintf('Min: %.3f', minVal), 'FontSize', 10);
        text(1.1, maxVal + ySpacing, sprintf('Max: %.3f', maxVal), 'FontSize', 10);
        text(1.1, meanVal, sprintf('Mean: %.3f', meanVal), 'FontSize', 10, 'Color', 'b');
        text(1.1, medVal, sprintf('Median: %.3f', medVal), 'FontSize', 10, 'Color', 'r');
        text(1.1, q1, sprintf('Q1: %.3f', q1), 'FontSize', 10);
        text(1.1, q3, sprintf('Q3: %.3f', q3), 'FontSize', 10);
        hold off;
    end
end

% Evaluate Combined Model
modelName = 'Combined';
all_R2 = zeros(repeats, 1);
all_MAE = zeros(repeats, 1);
all_RMSE = zeros(repeats, 1);

for r = 1:repeats
    cv = cvpartition(length(y), 'KFold', k);
    yTrue = [];
    yPred = [];

    for fold = 1:k
        trainIdx = training(cv, fold);
        testIdx = test(cv, fold);
        y_train = y(trainIdx);

        % Train Each Model
        % Random Forest
        rf_model = fitrensemble(Xmat(trainIdx, :), y_train, ...
            'Method', 'Bag', ...
            'NumLearningCycles', bestRF.NumLearningCycles, ...
            'Learners', rfTemplate);

        % Neural Network
        nn_model = fitrnet(Xmat_norm(trainIdx, :), y_train, ...
            'LayerSizes', layerSizes, ...
            'Lambda', bestNNstruct.Lambda, ...
            'Standardize', false);

        % Support Vector Machine
        svm_model = fitrsvm(Xmat_norm(trainIdx, :), y_train, ...
            'KernelFunction', kernelFunc, ...
            'BoxConstraint', bestSVM.BoxConstraint, ...
            'KernelScale', kernelScale, ...
            'Standardize', false);

        % === Predict and Average Each Model ===
        pred_rf = predict(rf_model, Xmat(testIdx, :));
        pred_nn = predict(nn_model, Xmat_norm(testIdx, :));
        pred_svm = predict(svm_model, Xmat_norm(testIdx, :));

        avg_pred = (pred_rf + pred_nn + pred_svm) / 3;
        yTrue = [yTrue; y(testIdx)];
        yPred = [yPred; avg_pred];
    end

    SSres = sum((yTrue - yPred).^2);
    SStot = sum((yTrue - mean(yTrue)).^2);
    all_R2(r) = 1 - SSres / SStot;
    all_MAE(r) = mean(abs(yTrue - yPred));
    all_RMSE(r) = sqrt(mean((yTrue - yPred).^2));
end

% Store Results
all_R2_all{end+1} = all_R2;
all_MAE_all{end+1} = all_MAE;
all_RMSE_all{end+1} = all_RMSE;

fprintf('\n--- %s ---\n', modelName);
fprintf('R² Mean: %.2f%%\n', mean(all_R2)*100);
fprintf('MAE Mean: %.3f\n', mean(all_MAE));
fprintf('RMSE Mean: %.3f\n', mean(all_RMSE));

metricNames = {'R² (%)', 'MAE', 'RMSE'};
rawMetrics = {all_R2 * 100, all_MAE, all_RMSE};

figure('Name', sprintf('%s - Evaluation Metrics', modelName), 'NumberTitle', 'off');
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for metricIdx = 1:3
    metricValues = rawMetrics{metricIdx};
    metricLabel = metricNames{metricIdx};

    q1 = quantile(metricValues, 0.25);
    q3 = quantile(metricValues, 0.75);
    minVal = min(metricValues);
    maxVal = max(metricValues);
    medVal = median(metricValues);
    meanVal = mean(metricValues);

    nexttile;
    boxplot(metricValues, 'Labels', {metricLabel});
    title(metricLabel);
    ylabel(metricLabel);
    grid on;

    hold on;
    ylims = ylim;
    ySpacing = (ylims(2) - ylims(1)) * 0.05;

    text(1.1, minVal - ySpacing, sprintf('Min: %.3f', minVal), 'FontSize', 10);
    text(1.1, maxVal + ySpacing, sprintf('Max: %.3f', maxVal), 'FontSize', 10);
    text(1.1, meanVal, sprintf('Mean: %.3f', meanVal), 'FontSize', 10, 'Color', 'b');
    text(1.1, medVal, sprintf('Median: %.3f', medVal), 'FontSize', 10, 'Color', 'r');
    text(1.1, q1, sprintf('Q1: %.3f', q1), 'FontSize', 10);
    text(1.1, q3, sprintf('Q3: %.3f', q3), 'FontSize', 10);
    hold off;
end

% Feature Importance Figure
figure('Units', 'normalized', 'Position', [0.15 0.15 0.7 0.75]);
t = tiledlayout(2,2, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% Categorical Features
nexttile(1);
bar(categoricalScores(categoryIdx), 'FaceColor', [0.2 0.6 0.8]);
title('Categorical Feature Group Importance', 'FontSize', 12);
ylabel('Importance', 'FontSize', 10);
xticks(1:length(categoryIdx));
xticklabels(sortedCategoricalGroups);
xtickangle(45);
ylim([0, max(categoricalScores)*1.15]);  % Stretch y for clarity
set(gca, 'FontSize', 9);
grid on;

% Numerical Features
nexttile(2);
bar(numericalScores(numIdx), 'FaceColor', [0.8 0.4 0.4]);
title('Numerical Feature Group Importance', 'FontSize', 12);
ylabel('Importance', 'FontSize', 10);
xticks(1:length(numIdx));
xticklabels(sortedNumericalGroups);
xtickangle(45);
ylim([0, max(numericalScores)*1.15]);
set(gca, 'FontSize', 9);
grid on;

% Overall Features (Categorical and Numerical)
nexttile([1 2]);
bar(groupScores(sortedIdxAll), 'FaceColor', [0.4 0.6 0.4]);
title('Overall Feature Group Importance', 'FontSize', 12);
ylabel('Importance', 'FontSize', 10);
xticks(1:length(sortedIdxAll));
xticklabels(groupNames(sortedIdxAll));
xtickangle(45);
ylim([0, max(groupScores)*1.15]);
set(gca, 'FontSize', 9);
grid on;