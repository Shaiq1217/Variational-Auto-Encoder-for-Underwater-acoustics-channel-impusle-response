% This program reads and plots extracted CIRs from lake experiment
% Created by Zheng Guo,University of Alabama
% Last updated Jul 20, 2020
% To read other cir files, change fn to the desinged cirmat file name

clear all; clc; close all;

% CIR extracted from lake tuscaloosa @Jul 9, 2019
fn='CIR@1610#1472'; % @ 1610 for #1472

%read the CIR measurements
load([fn,'.mat']);

%plot estimated CIRs in image
[a,b]=size(cirmat);
delay_vec=1e3*(0:b-1)/sampling_rate;
geo_vec=(0:a-1)*blk_dur;
figure;
imagesc(delay_vec,geo_vec,20*log10(abs(cirmat)));
axis ij; set(gca,'Fontsize',16,'FontWeight','bold');
axis([0,40,0,max(geo_vec)])
% add colorbar
% title(fn)
shading interp;colormap jet
h=colorbar;
set(h,'Fontsize',14,'FontWeight','bold'); title(h,'Rel. dB');
caxis([120 160]);
set(gca, 'FontName', 'times')
xlabel('Delay (ms)')
ylabel('Geo-time (s)')