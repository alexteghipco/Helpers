function [mn,vals,subs,roiNms] = extractROIsFromMatFiles(inDir,fld,roiIds)
%% [mn,vals,subs,rois] = extractROIsFromMatFiles(inDir,fld,method,rois)
%
% Example call: [mn,vals,subs,rois] = extractROIsFromMatFiles('/Users/alex/test','rest_jhu',[]);
%
% This function loops through a directory (inDir) containing .mat files of
% subject-level neuroimaging data and extracts values from specific rois
% (roiIds) contained in a specific field (i.e., modality + atlas; fld).
%
% The function returns these values for each subject (vals), but also a
% mean value, either across subjects or across ROIs (mn). This also works
% for connectomes stored in the .mat files. For functional
% connectivity-based connectomes, the retrieved values will include r and p
% values. The output variable vals will be an n x p matrix for
% non-connectome structures. Here, n refers to the number of participants
% and p to the number of ROIs you specified to extract. You will find ROI
% names in the p x 1 vector output as roiNms, and you will find subject
% names in the n x 1 vector output as subs. Mean values will be a p x 1
% vector. For structural connectome data, vals will be a 3D matrix that is
% p x p x n, representing connections between ROIs p and p, for each
% participant n. The mean values in mn will be p x p, collapsing across
% subjects. For functional connectome data, vals will be a 4D matrix that
% is p x p x n x 2, representing connections between ROIs p and p, for each
% participant n. The fourth dimension corresponds to r and p values. p x p
% x n x 1 represents all of the r values whereas p x p x n x 2 represents
% all of the p values.
%
% Inputs: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% inDir -- is a path to a directory containing only .mat files with
% subject-level data (i.e., it can contain other file types, but no random
% .mat files). It *should* be okay if some participants don't have the fld
% name supplied (see fld below).
%
% fld -- field within the .mat file structure that you would like to
% analyze supplied as a string (e.g., 'dtifc_AICHA', 'rest_jhu', etc). If
% you don't know which field you would like to look at, just load one of
% the .mat files into MATLAB and find one you like.
% As in: 
% >> tmp = load('ABC1002.mat');
% >> fldsToSelectFrom = fieldnames(tmp)
%
% roiIds -- supply ROIs to analyze. Leave empty (i.e., []) to analyze all
% rois. Otherwise provide an array of numbers that correspond to ROI
% indices (e.g., 1 for JHU is L_SFG, etc.)
%
% Outputs: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% mn --
% mn -- mean of values extracted. If non-connectome data, mean is computed
% across ROIs (i.e., 1 mean value per participant), returning an n x 1
% vector where n is the number of participants. If connectome data, you get
% a value per connection, reflecting mean connection strength across
% participants. In this case, the returned mn matrix will be p x p,
% representing connections between ROI pairs.
%
% vals -- values extracted. If non-connectome data, it is n x p where n is
% number of subjects and p is numbr of ROIs. If structural connectome, it
% is p x p x n where p x p represents connections between ROI pairs. If
% functional connectome, it is p x p x n x 2 (p x p x n x 1 is r-values
% between ROI pairs for each subject; p x p x n x 2 is p-values between ROI
% pairs for each subject).
%
% subs -- subject names corresponding to n in vals and/or mn.
%
% roiNms -- roi names corresponding to p in vals and/or n.
%
% 
%% Alex Teghipco // alex.teghipco@sc.edu

% Do some basic error checks first...
v = version('-release');
if str2double(v(1:end-1)) < 2016 || (str2double(v(1:end-1)) == 2016 && strcmpi(v(end),a))
    error('Please update to at matlab 2016b or later releases...')
end

if isunix 
    s = '/';
else
   s = '\'; 
end

id = strfind(fld,'_');
if isempty(id) 
    error(['You cannot analyze this structure, it likely contains voxelwise data: ' fld])
end

% separate atlas and modality 
fldnm = fld(1:id-1);
atls = fld(id+1:end);

% get mat files
d = dir([inDir s '*.mat']);

% get roi ids within atlas if not provided...we loop over participants in
% case some mat files don't contain the data you are interesting in
% analyzing...
if isempty(roiIds)
    for i = 1:length(d)
        try
            tmp = load([inDir s d(i).name]);
            tmps = tmp.(fld); % extract field you want to analyze...
            c = cellstr(tmp.(fld).label); % convert to make extractBefore work--faster than the alternative methods I tried....
            roiIds = str2double(extractBefore(c,'|')); % find each roi id for atlas
        catch
            disp(['Searching a new subject for user-specified fieldname from which we can extract ROI ids...']);
        end
        if ~isempty(roiIds)
            break
        end
    end
end

% setup output structure for values
incp = 0;
if strcmpi(fldnm,'rest')
    vals = nan(length(roiIds),length(roiIds),length(d),2);
    incp = 1;
elseif strcmpi(fldnm,'dti') || strcmpi(fldnm,'dtifc') || strcmpi(fldnm,'dtimx') || strcmpi(fldnm,'dtimn')
    vals = nan(length(roiIds),length(roiIds),length(d));
else
    vals = nan(length(d),length(roiIds));
end

% main loop
for i = 1:length(d)
    disp(['Working on subject: ' d(i).name ' ... ' num2str(length(d)-i) ' mat files left'])
    tmp = load([inDir s d(i).name]);
    try
        tmps = tmp.(fld); % extract field you want to analyze...
        c = cellstr(tmp.(fld).label); % convert to make extractBefore work--faster than the alternative methods I tried....
        atlsIds = str2double(extractBefore(c,'|')); % find each roi id for atlas
        [~,ia,~] = intersect(atlsIds,roiIds); % find your rois in atlas
        if ~exist('roiNms','var') % get roi names if we don't already have them
            roiNms = c(ia);
        end
        
        if ~isfield(tmps,'r')
            vals(i,:) = tmp.(fld).mean(ia);
        else
            if incp
                vals(:,:,i,1) = tmp.(fld).r(ia,ia);
                vals(:,:,i,2) = tmp.(fld).p(ia,ia);
            else
                vals(:,:,i) = tmp.(fld).r(ia,ia);
            end
        end
    catch
        disp(['FAILURE...most likely could not find the structure you have specified for participant.']);
    end
    subs{i,1} = d(i).name;
end

% get means of values
if length(size(vals)) == 2
    mn = mean(vals,2,'omitnan');
elseif length(size(vals)) == 3
    mn = mean(vals,3,'omitnan');
elseif length(size(vals)) == 4
    mn = mean(vals(:,:,:,1),3,'omitnan');
end
