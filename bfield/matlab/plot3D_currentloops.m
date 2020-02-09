function plot3D_currentloops( CurrentLoops, nSamplePoints, FigureID )

figure( FigureID )

for ii = 1 : size(CurrentLoops,1)
    
    % Current Loop of index 'ii'
    OC_LAB  =  CurrentLoops( ii, 1:3 )';
    nh      =  CurrentLoops( ii, 4:6 )';
    I0      =  CurrentLoops( ii,  7  );
    Ra      =  CurrentLoops( ii,  8  );
    Nw      =  CurrentLoops( ii,  9  );
    
    % Generate nSamplePoints loop points in the (x,y) plane
    dz = 2.0*pi/nSamplePoints;
    cz = cos( 0.0 : dz : dz*nSamplePoints);
    sz = sin( 0.0 : dz : dz*nSamplePoints);
    
    LoopX =   Ra * cz;
    LoopY =   Ra * sz;
    LoopZ =  0.0 * cz;
    
    % Roto-translation of the Loop points
    ROT_LAB_LOOP = roto( nh );
    
    for jj=1:numel(sz)
        CurrentLoopX(jj) = OC_LAB(1) + ROT_LAB_LOOP(1,:) * [ LoopX(jj); LoopY(jj); LoopZ(jj) ];
        CurrentLoopY(jj) = OC_LAB(2) + ROT_LAB_LOOP(2,:) * [ LoopX(jj); LoopY(jj); LoopZ(jj) ];
        CurrentLoopZ(jj) = OC_LAB(3) + ROT_LAB_LOOP(3,:) * [ LoopX(jj); LoopY(jj); LoopZ(jj) ];
    end
    
    % plot 3D loop center
    plot3( OC_LAB(1), OC_LAB(2), OC_LAB(3), 'ko'); 
    hold on
    
    % plot 3D loop coil
    plot3( CurrentLoopX, CurrentLoopY, CurrentLoopZ, 'k*-' )
    
end

grid on
axis equal

xlabel('X')
ylabel('Y')
zlabel('Z')


