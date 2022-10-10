function [roiNm,roiData,fl] = extractPerctDmgNormLsn(atlsNm)
% Script to facilitate extracting % damage in each ROI within an atlas.
% Uses NiiStat's nii_roi2stats.m function. atlsNm is the string of any
% atlas in your NiiStat directory (e.g., 'jhu'). Script will pull up a
% prompt to select all subject-level normalized lesion mask files (must be
% .nii) for which you would like to extract % damage. Output roiNm is a
% char array with all roi names. These names map onto the rows of roiData.
% Columns of roiData contain each subject. Subject identity is stored in
% cell array fl. Values in roiData describe proportion of ROI that is in
% the lesion mask (i.e., damaged). 
%
% Please ensure NiiStat is in your MATLAB path before running.
%
% Example call: [roiNm,roiData,fl] = extractPerctDmgNormLsn('jhu');
% You can omit passing in a specific atlas name and this function will
% automatically use jhu: [roiNm,roiData,fl] = extractPerctDmgNormLsn([]);
%
% Alex Teghipco // 10/10/22

warning('Please ensure NiiStat is in your MATLAB path')

if isempty(atlsNm)
    atlsNm = 'jhu';
end

[fl,pth] = uigetfile({'*.nii'},'Please select subject-level normalized lesion file(s)','MultiSelect','on');
if ~iscell(fl)
    fl = {fl};
end

for i = 1:length(fl)
    s = nii_roi2stats(atlsNm,[pth fl{i}],'','lesion_'); %3d volume
    if i == 1
       roiNm = s.(['lesion_' atlsNm]).label;
    end
    roiData(:,i) = s.(['lesion_' atlsNm]).mean;
end
