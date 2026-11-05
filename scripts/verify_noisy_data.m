clc;
clear;

root = fullfile('data', 'noisy');

files = dir(fullfile(root, '**', '*.mat'));

fprintf('\nDATASET INVENTORY\n');
fprintf('MAT files found: %d\n', numel(files));

if numel(files) ~= 90
    warning('Expected 90 datasets, but found %d.', numel(files));
end

if isempty(files)
    error('No MAT files found. Check the output directory.');
end

% Inspect every dataset
for k = 1:numel(files)

    filepath = fullfile(files(k).folder, files(k).name);
    info = whos('-file', filepath);

    if k == 1
        fprintf('\nVARIABLES IN FIRST DATASET:\n');

        for j = 1:numel(info)
            fprintf('%-20s ', info(j).name);
            fprintf('%s\n', mat2str(info(j).size));
        end
    end

    % Check that datasets contain the expected variables
    names = {info.name};

    required = {'t_all', 'x_all'};

    missing = setdiff(required, names);

    if ~isempty(missing)
        fprintf('Missing variables in: %s\n', files(k).name);
    end

end

fprintf('\nInventory completed.\n');