clear sp spp M

% MP=readC3D('01_01.c3d'); % playground - forward jumps, turn around
% MP=readC3D('01_02.c3d');
% MP=readC3D('01_09.c3d'); %playground - climb, hang, hold self up with arms straight, swing, drop, sit, dangle legs, go under
MP=readC3D('02_01.c3d'); % walk
% MP=readC3D('02_03.c3d'); % run/jog

MP=MP(:,1:41,:);

[frames, markers,~]=size(MP);

hf=figure();
ax=gca;

axis equal;
view([60,20]);
ax.XLimMode='manual';
ax.YLimMode='manual';
ax.ZLimMode='manual';
ax.XLim=[-500 500];
ax.YLim=[-500 500];
ax.ZLim=[-1000 1000];

mode=4; % 1: jump, 4: walk

no=1;
for i=1:1:frames
    [spp(:,:,no),pos]=generateSP2(MP(i,:,1),MP(i,:,2),MP(i,:,3));
    no=no+1;

    figure(1)
    scatter3(pos(:,1), pos(:,2),pos(:,3),70,'o','filled','blud');

    t=title(num2str(1),'FontSize',24,'Color',"k");
    switch mode
        case 1
            drawlines(pos,[9 1 2 34 6 14 21 26 33 31 28 26]);
            drawlines(pos,[38 36 18 12 35 5 7 11 8]);
            drawlines(pos,[10 1 3 4 13 20 27 29 30 32 27]);
            drawlines(pos,[18 12 35 5 7 11 8]);
            drawlines(pos,[38 37 15 17 19 41 40 39]);
            drawlines(pos,[10 9 16 38]);
            drawlines(pos,[24 25 22 23]);
        case 4
            drawlines(pos,[1 2 4 11 22 26 32 33 34 39 32]);
            drawlines(pos,[13 38 21 35 20 9 7 17 16 6]);
            drawlines(pos,[1 3 19 25 30 36 37 40 41 36]);
            drawlines(pos,[14 38 23 18 12 5 8 15 10]);
            drawlines(pos,[1 13 14]);
            drawlines(pos,[1 14]);
            drawlines(pos,[27 28 29]);
            drawlines(pos,[31 28 24 38]);        
    end

    t.String=num2str(i);
    % for j=1:size(pos,1)
    %     text(pos(j,1),pos(j,2),pos(j,3),num2str(j),'FontSize',16);
    % end
    
    ax=gca;
    axis equal;
    ax.XTickLabel = []; % X軸TickLabelに空を設定
    ax.YTickLabel = []; % Y軸TickLabelに空を設定
    ax.ZTickLabel = []; % Z軸TickLabelに空を設定
    view([60,20]);
    ax.XLimMode='manual';
    ax.YLimMode='manual';
    ax.ZLimMode='manual';
    ax.XLim=[-500 500];
    ax.YLim=[-500 500];
    ax.ZLim=[-1000 1000];

    pause(0.05);
    if strcmp(get(hf,'currentcharacter'),'q')
        close(hf);
        break
    end
    hold off
end

sp=spp;

function [] = drawlines(pos,no)
line([pos(no,1)],[pos(no,2)],[pos(no,3)],'LineStyle','-','LineWidth',2,'Color','blue');
end

function [sp, pos] = generateSP2(x,y,z)

dim=size(x,2);
pos=[x' y' z'];

gpos=mean(pos,1);
pos=pos-ones(dim,3)*diag(gpos);

sp=orth(pos);
end