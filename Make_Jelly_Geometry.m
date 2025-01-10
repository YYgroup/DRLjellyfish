%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Jellyfish Geometry modified based on IB2d jellyfish example
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Make_Jelly_Geometry()

close all;
clear all;
Lfactor=0.1;
L = 8*Lfactor;                              % height of computational domain (m) for keeping desired resolution
Lh = 16*Lfactor;                            % actual height of computational domain (m) (MATCHES INPUT2D)
Lw = 16*Lfactor;                             % width of computational domain (m) (MATCHES INPUT2D)
N = 256;                            % number of Cartesian grid meshwidths at the finest level of the AMR grid
dx = L/N;                           % Cartesian mesh width (m)
ds = dx/2;

rho=100; 

U=f*Lfactor;
mesh_name = 'jelly';

xShift = Lw/2;
yShift = Lh/4;

a=0.5*Lfactor;                               % bell radius (semi-minor axis, horizontal axis, note width=2a)
b=0.75*Lfactor;                              % bell semi-major axis 
d=-0.25*Lfactor;                             % y-length of muscle


x_points=zeros(1000,1);
z_points=zeros(1000,1);

theta=zeros(1000,1);
theta_lim=asin(d/b);
theta_test=pi/2;
 

id_points=zeros(1000,1);


kappa_spring = 1.0563*rho*U^2*Lfactor*springFactor; %1e5               % spring constant (Newton)
kappa_beam = 0.0528*rho*U^2*Lfactor^-1; %78.8699;%52.8*0.86^2; %2.5e5                % beam stiffness constant (Newton m^2)

c=0;
while(theta_test<(pi-theta_lim))
    c=c+1;
    theta(c)=theta_test;
     
    x_points(c)=a*cos(theta(c));
    z_points(c)=b*sin(theta(c));
    id_points(c)=c-1;
     
    theta_test=ds/((a*sin(theta(c)))^(2)+(b*cos(theta(c)))^(2))^(.5)+theta(c);
     
end
 
npts=2*c-1;
npts_wing=floor(npts/2);
npts_musc=floor(npts_wing/4);
 
for j=(c+1):(npts)
    x_points(j)=-1*x_points(j-c+1);
    z_points(j)=z_points(j-c+1);
    id_points(j)=j-1;
end
 
x_points=x_points(1:npts)+xShift;
z_points=z_points(1:npts)+yShift;

plot(x_points(:),z_points(:),'*'); hold on;
axis([0 Lw 0 Lh])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Print .vertex information
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vertex_fid = fopen([mesh_name num2str(N) '.vertex'], 'w');
 
    fprintf(vertex_fid, '%d\n', npts);
    lag_ct = 0;
    
    %
    % bell
    %
    factor=1;
    for j=1:npts
        %fprintf(vertex_fid, '%1.16e %1.16e\n', x_points(j), z_points(j));
        fprintf(vertex_fid, '%1.16e %1.16e\n', x_points(j)*factor+xShift*(1-factor), z_points(j));
        lag_ct = lag_ct + 1;
    end
    %
    % muscles
    %
    for s = 1:npts_musc
        %fprintf(vertex_fid, '%1.16e %1.16e\n', x_points(npts_wing+1-npts_musc+s), z_points(npts_wing+1-npts_musc+s));
        plot(x_points(npts_wing+1-npts_musc+s),z_points(npts_wing+1-npts_musc+s),'r*'); hold on;
        lag_ct = lag_ct + 1;
    end
    for s = 1:npts_musc
        %fprintf(vertex_fid, '%1.16e %1.16e\n', x_points(npts-npts_musc+s), z_points(npts-npts_musc+s));
        plot(x_points(npts-npts_musc+s),z_points(npts-npts_musc+s),'r*'); hold on;
        lag_ct = lag_ct + 1;
    end
    
fclose(vertex_fid);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Print .spring information
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spring_fid = fopen([mesh_name num2str(N) '.spring'], 'w');
    
    npts_spring_type1=npts-1;
 
    fprintf(spring_fid, '%d\n', npts-1);%+ npts_musc
 
    fprintf('\nNumber of springs before muscles: %d \n\n',npts-1);
    
    factor = 1;%ds^2/ds; %1
    
    %
    % bell
    %
    for s = 1:c-1
        resting=sqrt((x_points(s)-x_points(s+1))^(2)+(z_points(s)-z_points(s+1))^(2));
        fprintf(spring_fid, '%d %d %1.16e %1.16e\n', id_points(s), id_points(s+1), kappa_spring*ds/(ds^2)*factor, resting);
    end
    for s = c+1:npts-1
        resting=sqrt((x_points(s)-x_points(s+1))^(2)+(z_points(s)-z_points(s+1))^(2));
        fprintf(spring_fid, '%d %d %1.16e %1.16e\n', id_points(s), id_points(s+1), kappa_spring*ds/(ds^2)*factor, resting);
    end
    resting=sqrt((x_points(1)-x_points(c+1))^(2)+(z_points(1)-z_points(c+1))^(2));
    fprintf(spring_fid, '%d %d %1.16e %1.16e\n', id_points(1), id_points(c+1), kappa_spring*ds/(ds^2)*factor, resting);

    %
    % muscles
    %
    %for s = 1:npts_musc
    %    fprintf(spring_fid, '%d %d %1.16e %1.16e\n',id_points(npts_wing+1-npts_musc+s), id_points(npts-npts_musc+s), F, abs(x_points(npts_wing+1-npts_musc+s)-x_points(npts-npts_musc+s)));
    %end
 
    fclose(spring_fid);
 
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Print .beam information
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beam_fid = fopen([mesh_name num2str(N) '.beam'], 'w');
    % need to change to torsional spring
    fprintf(beam_fid, '%d\n', npts-2);
    for s = c-1:-1:2
        C = (x_points(s-1)-x_points(s))*(z_points(s)-z_points(s+1))-(z_points(s-1)-z_points(s))*(x_points(s)-x_points(s+1));
        C1 = x_points(s-1)+x_points(s+1)-2*x_points(s);
        C2 = z_points(s-1)+z_points(s+1)-2*z_points(s);
        fprintf(beam_fid, '%d %d %d %1.16e %1.16e %1.16e\n', id_points(s+1), id_points(s), id_points(s-1), kappa_beam*ds/(ds^2), C,C2);
    end

    C = (x_points(c+1)-x_points(1))*(z_points(1)-z_points(2))-(z_points(c+1)-z_points(1))*(x_points(1)-x_points(2));
    C1 = x_points(c+1)+x_points(2)-2*x_points(1);
    C2 = z_points(c+1)+z_points(2)-2*z_points(1);
    fprintf(beam_fid, '%d %d %d %1.16e %1.16e %1.16e\n', id_points(2),id_points(1), id_points(c+1), kappa_beam*ds/(ds^2), C,C2);
    
    C = (x_points(c+2)-x_points(c+1))*(z_points(c+1)-z_points(1))-(z_points(c+2)-z_points(c+1))*(x_points(c+1)-x_points(1));
    C1 = x_points(c+2)+x_points(1)-2*x_points(c+1);
    C2 = z_points(c+2)+z_points(1)-2*z_points(c+1);
    fprintf(beam_fid, '%d %d %d %1.16e %1.16e %1.16e\n', id_points(1),id_points(c+1), id_points(c+2), kappa_beam*ds/(ds^2), C,C2);
    
    for s = c+2:npts-1
        C = (x_points(s+1)-x_points(s))*(z_points(s)-z_points(s-1))-(z_points(s+1)-z_points(s))*(x_points(s)-x_points(s-1));
        C1 = x_points(s+1)+x_points(s-1)-2*x_points(s);
        C2 = z_points(s+1)+z_points(s-1)-2*z_points(s);
        %fprintf(beam_fid, '%d %d %d %1.16e %1.16e %1.16e\n', id_points(s-1), id_points(s), id_points(s+1), kappa_beam*ds/(ds^4), C1,C2);
        fprintf(beam_fid, '%d %d %d %1.16e %1.16e %1.16e\n', id_points(s-1), id_points(s), id_points(s+1), kappa_beam*ds/(ds^2), C,C2);
    end
    
    fclose(beam_fid);
%print context for RL
fid=fopen('context_for_RL.txt', 'w');
fprintf(fid,'points of jellywing = %d\npoints of muscle = %d\nmuscle spring ID starts from %d\ntotal lag pointes = %d\nbeam_num = %d',npts_wing,npts_musc,npts-1,npts,npts-2);
fclose(fid);

