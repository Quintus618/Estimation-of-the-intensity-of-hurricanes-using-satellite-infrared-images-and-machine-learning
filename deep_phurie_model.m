clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROGRAM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hurricane localized in the sea
nc_file_1 = './HURSAT-B1/2004/HURSAT_b1_v06_2004247N10332_IVAN_c20170721/2004247N10332.IVAN.2004.09.09.2100.18.GOE-12.114.hursat-b1.v06.nc';

% Hurricane where its image has zero pixel intensity (landfall)
nc_file_2 = './HURSAT-B1/2005/HURSAT_b1_v06_2005236N23285_KATRINA_c20170721/2005236N23285.KATRINA.2005.08.26.0600.66.GOE-10.057.hursat-b1.v06.nc';

% Hurricen localized after landfall
nc_file_3 = './HURSAT-B1/2005/HURSAT_b1_v06_2005236N23285_KATRINA_c20170721/2005236N23285.KATRINA.2005.08.29.1500.39.GOE-12.084.hursat-b1.v06.nc';

% IR image with NaN
nc_file_4 = './HURSAT-B1/2004/HURSAT_b1_v06_2004260N11331_KARL_c20170721/2004260N11331.KARL.2004.09.18.0600.46.GOE-12.075.hursat-b1.v06.nc';

% sea
nc_file_5 = './HURSAT-B1/2004/HURSAT_b1_v06_2004260N11331_KARL_c20170721/2004260N11331.KARL.2004.09.19.1800.41.GOE-12.097.hursat-b1.v06.nc';

nc_file_6 = './HURSAT-B1/2007/HURSAT_b1_v06_2007244N12303_FELIX_c20170721/2007244N12303.FELIX.2007.09.04.0300.18.GOE-12.110.hursat-b1.v06.nc';

rgb = [ ...
    94    79   162
    50   136   189
   102   194   165
   171   221   164
   230   245   152
   255   255   191
   254   224   139
   253   174    97
   244   109    67
   213    62    79
   158     1    66  ] / 255;

% Get information about the file
ncinfo(nc_file_1)

% Display the file
ncdisp(nc_file_1);

% Open and read a NC file with particular variable
% satellite IR image of a hurricane where we apply a colormap
hurricane_IR_image_1 = ncread(nc_file_1,'IRWIN');
hurricane_CO_image_1 = ncread(nc_file_1,'IRCO2');
imshow(hurricane_IR_image_1);
title("IVAN Hurricane IR")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_1 = ncread(nc_file_1,'VSCHN');
imshow(hurricane_visible_image_1);
title("IVAN Hurricane visible")
figure


% Get some precise information about nc file
hurricane_wind_speed_1 = ncread(nc_file_1,'WindSpd');
hurricane_long_cent_1 = ncread(nc_file_1,'archer_lon');
hurricane_lat_cent_1 = ncread(nc_file_1,'archer_lat');
hurricane_sat_name_1 = ncreadatt(nc_file_1,"/","Satellite_Name");

hurricane_IR_image_2 = ncread(nc_file_2,'IRWIN');
hurricane_CO_image_2 = ncread(nc_file_2,'IRWVP');
imshow(hurricane_IR_image_2);
title("KATRINA Hurricane IR 1")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_2 = ncread(nc_file_2,'VSCHN');
imshow(hurricane_visible_image_2);
title("KATRINA Hurricane visible 1 zero pixel intensity")
figure

hurricane_wind_speed_2 = ncread(nc_file_2,'WindSpd');
hurricane_long_cent_2 = ncread(nc_file_2,'archer_lon');
hurricane_lat_cent_2 = ncread(nc_file_2,'archer_lat');
hurricane_sat_name_2 = ncreadatt(nc_file_2,"/","Satellite_Name");

hurricane_IR_image_3 = ncread(nc_file_3,'IRWIN');
hurricane_CO_image_3 = ncread(nc_file_3,'IRCO2');
imshow(hurricane_IR_image_3);
title("KATRINA Hurricane IR landfall")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_3 = ncread(nc_file_3,'VSCHN');
imshow(hurricane_visible_image_3);
title("KATRINA Hurricane visible landfall")
figure

hurricane_wind_speed_3 = ncread(nc_file_3,'WindSpd');
hurricane_long_cent_3 = ncread(nc_file_3,'archer_lon');
hurricane_lat_cent_3 = ncread(nc_file_3,'archer_lat');
hurricane_sat_name_3 = ncreadatt(nc_file_3,"/","Satellite_Name");

hurricane_IR_image_4 = ncread(nc_file_4,'IRWIN');
imshow(hurricane_IR_image_4);
title("KEN Hurricane IR landfall")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_4 = ncread(nc_file_4,'VSCHN');
imshow(hurricane_visible_image_4);
title("KEN Hurricane visible landfall")

hurricane_wind_speed_4 = ncread(nc_file_4,'WindSpd');
hurricane_long_cent_4 = ncread(nc_file_4,'archer_lon');
hurricane_lat_cent_4 = ncread(nc_file_4,'archer_lat');
hurricane_sat_name_4 = ncreadatt(nc_file_4,"/","Satellite_Name");

hurricane_IR_image_5 = ncread(nc_file_5,'IRWIN');
hurricane_CO_image_5 = ncread(nc_file_5,'IRCO2');
imshow(hurricane_IR_image_5);
title("KEN Hurricane IR landfall")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_5 = ncread(nc_file_5,'VSCHN');
imshow(hurricane_visible_image_5);
title("KEN Hurricane visible landfall")
figure

hurricane_wind_speed_5 = ncread(nc_file_5,'WindSpd');
hurricane_long_cent_5 = ncread(nc_file_5,'archer_lon');
hurricane_lat_cent_5 = ncread(nc_file_5,'archer_lat');
hurricane_sat_name_5 = ncreadatt(nc_file_5,"/","Satellite_Name");

hurricane_IR_image_6 = ncread(nc_file_6,'IRWIN');
hurricane_CO_image_6 = ncread(nc_file_6,'IRCO2');
imshow(hurricane_IR_image_6);
title("FELIX Hurricane IR")
colormap(rgb);
clim([200 320]);
colorbar;
figure

hurricane_visible_image_6 = ncread(nc_file_6,'VSCHN');
imshow(hurricane_visible_image_6);
title("FELIX Hurricane visible")


% Get some precise information about nc file
hurricane_wind_speed_6 = ncread(nc_file_6,'WindSpd');
hurricane_long_cent_6 = ncread(nc_file_6,'archer_lon');
hurricane_lat_cent_6 = ncread(nc_file_6,'archer_lat');
hurricane_sat_name_6 = ncreadatt(nc_file_6,"/","Satellite_Name");

% Hurricane contours
%annotated_hurricane_center(hurricane_visible_image_3, 50, 'c.', [5000,6000,7000,8000,9000,10000], 5)

% Eye of the hurricane
%annotated_hurricane_center(hurricane_IR_image_1, 5, 'r.', [500,1000,1500,2000,3500,4000], 2)

% Test detection of zero pixel intensity and negative pixels
%detected_1 = pixel_treatment(hurricane_IR_image_1);
%detected_2 = pixel_treatment(hurricane_IR_image_2);
%detected_3 = pixel_treatment(hurricane_IR_image_3);
%detected_4 = pixel_treatment(hurricane_IR_image_4); % problem
%detected_5 = pixel_treatment(hurricane_IR_image_5);
detected_6 = pixel_treatment(hurricane_IR_image_5);

d_1 = remove_landfall(hurricane_CO_image_1);
%d_2 = remove_landfall(hurricane_CO_image_2);
d_3 = remove_landfall(hurricane_CO_image_3);
%d_4 = remove_landfall(hurricane_IR_image_4);
d_5 = remove_landfall(hurricane_CO_image_5);
d_6 = remove_landfall(hurricane_CO_image_6);

% Convolutional neural network trained
%h5_file = './deep-Phurie-master/model/model.h5'
%h5disp(h5_file)

% Download HURSAT-B1 dataset from 2004 to 2009
%download_HURSAT_B1("https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/", 2004, 2009, "./HURSAT_B1/")

% Filter the dataset (preprocessing)
%preprocessing()

% Split into training set and test set


% Model
%[layers, options] = convo_neural_network()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS AND PROCEDURES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Download HURSAT-B1 dataset from the url
% Create HURSAT-B1 folder
% Inside this folder, create a folder for each year from 2004 to 2009
% HURSAT-B1
% => 2004 2005 2006 2007 2008 2009
% Fill each folder with respective hurricanes data
% Extract hurricane data
% WARNING : the size of the dataset is 21Go
% example usage :
% download_HURSAT_B1("https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/",
%                       2004, 2009, "C:\Users\...\");
function download_HURSAT_B1(base_url, base_year, last_year, folder_path)
    %base_url="https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/";
    %base_year = 2004;
    years_apart = last_year - base_year;

    %folder_path = "C:\Users\momop\Travail\Poliba\image_processing\projet\dataset\";
    % TODO check if file already downloaded

    % creates the base folder of the dataset if it doesn't already exists 
    if ~exist(strcat(folder_path, pathsep, 'HURSAT-B1'))
        mkdir(folder_path, 'HURSAT-B1');
    end
    folder_path = strcat(folder_path, pathsep, 'HURSAT-B1');

    for i = 0:years_apart
        year = base_year + i;
        % creates a folder for each year
        if ~exist(strcat(folder_path, pathsep, num2str(year)), 'dir')
            mkdir(folder_path, num2str(year));
        end
    
        %fprintf('it=%i\n', i);
        
        % reads the raw content of the dataset webpage
        url=strcat(base_url, num2str(year));
        url=strcat(url, '/');
        raw = webread(url);
    
        % extracts each download link on the webpage
        out = regexp(raw, '<a href="HURSAT[^<]*\.tar\.gz">', "match");
        for j=1:length(out)
            % downloads each archive
            current_file = char(extractBetween(out{j}, '<a href="', '">'));
            file_url = strcat(url, current_file);
            save_path = strcat(folder_path, num2str(year));
            save_path = strcat(save_path, pathsep);
            save_path = strcat(save_path, current_file);
            websave(save_path, file_url)

            % creates a folder for each hurricane
            current_folder = strcat(folder_path, num2str(year), pathsep, extractBefore(current_file, ".tar.gz"));
            if ~exist(current_folder, 'dir')
                mkdir(current_folder);
            end
            % extracts the content of each archive
            untar(save_path, current_folder);
            % deletes each archive once extracted to free the space
            delete(save_path);
        end
    end
end

% Annotate the center of the hurricane
% Box that indicates the center of the hurricane
function annotated_hurricane_center(image, radius, color, iter, sizemark)
    
    [M,N] = size(image);

    % Initialisation
    x0 = round(M/2);
    y0 = round(N/2);
    r = radius;         %radius
    phi0 = levelsetFunction('circular',M,N,x0,y0,r);
    figure, imshow(image)
    hold on
    c = contourc(phi0,[0,0]);
    curveDisplay(c(2,:),c(1,:),color,'MarkerSize',sizemark)
    hold off

    % Iterate 
    iterations = iter;
    mu = 2.0;
    nu = 0.0;
    lambda1 = 1;
    lambda2 = 1;

    for K = 1:numel(iterations)
        phi = phi0;
        C = 0.5;
        niter = iterations(K);
        for I = 1:niter
            F = levelsetForce('chanvese', {image,phi,mu,nu,lambda1,lambda2},{'Fn','Cn'});
            phi = levelsetIterate(phi,F,0.5);

            if ~rem(I,5)
                phi = levelsetReset(phi,5,0.5);
            end
        end

        c = contourc(phi, [0 0]);
        figure, imshow(image)
        hold on
        curveDisplay(c(2,:),c(1,:),color,'MarkerSize',sizemark)
        hold off
    end
end

% Detect and remove landfall image
% return True if landfall is detected
function detected = remove_landfall(image)
    meanIntensity = mean(image(:));
    stdIntensity = std(image(:));
    disp("Mean: " + meanIntensity)
    disp("Standard deviation: " + stdIntensity) % interesting point to detect landfall
    if stdIntensity <= 20.4
        detected = true
    else
        detected = false
    end
end

% Detect and remove zero pixel intensity and negative pixel in an image
% if detected, return True
function detected = pixel_treatment(image_IR)
    min_visible = min(image_IR(:));
    test_NaN = anynan(image_IR);
    disp("Valeur min: " + min_visible)
    disp("Présence NaN: " + test_NaN)
    if min_visible <= 120 || test_NaN == 1
        detected = true;
    else
        detected = false;
    end
end

% preprocessing phase
function preprocessed_data = preprocessing()

    number_good_image = 0;

    % Browse HURSAT-B1 folders and browse each hurricane folder
    folder_path = '\'
    %{
    for year = 2004:2009
        if exist(strcat(folder_path, num2str(year)), 'dir')
            year_folders = dir
        end
    end
    %}
    
    % get each year folders
    year_folders = ls;
    for element in year_folders
        % checking if element is a folder
        if exist(element, 'dir')
            
        end
    end

    % For each file, apply the preprocessing

        hurricane_IR_image = ncread(nc_file,'IRWIN');

        % remove images with zero pixel intensity and negative pixels
        detected = pixel_treatment(hurricane_IR_image)
        if detected == false
            number_good_image = number_good_image + 1;
        else 
            continue;
        end
        end
    
        % remove images with landfall
        detected_land = remove_landfall(hurricane_IR_image)
        if detected == false
            number_good_image = number_good_image + 1;
        else
            continue;
        end
        end
    
        % resize image
        image_IR = imresize(hurricane_IR_image, [224, 224]);

    % print for each year, the number of good images
    disp("Year: " + year)
    disp("Number of good images:" + number_good_image)

    
        % add image to the dataset
end

% Convolutional Neural Network
function [layers, options] = convo_neural_network()
    layers = [
        imageInputLayer([224, 224, 1])
        convolution2dLayer(5, 32)
        maxPooling2dLayer(5, 'Stride', 2)
        convolution2dLayer(3, 64)
        maxPooling2dLayer(3, 'Stride', 2)
        convolution2dLayer(3, 64)
        maxPooling2dLayer(3, 'Stride', 1)
        convolution2dLayer(3, 64)
        maxPooling2dLayer(3, 'Stride', 1)
        convolution2dLayer(3, 128)
        maxPooling2dLayer(3, 'Stride', 1)
        convolution2dLayer(3, 128)
        maxPooling2dLayer(3, 'Stride', 1)
        fullyConnectedLayer(512)
        fullyConnectedLayer(64)
    ]
    options = trainingOptions('adam', 'OutputFcn', 'TrainingRMSE', 'MaxEpochs', 1000);
end

% TODO : Hyperparameter tuning for the CNN



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% IMPORTED LIBRARY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 % A COMMENTER
        function phi = levelsetFunction(type,varargin)
            %LEVELSETFUNCTION Generates a level-set function.
            %   PHI = LEVELSETFUNCTION(TYPE,VARARGIN) generates a level set
            %   (signed distance) function, PHI.  
            %
            %   If TYPE = 'mask' and VARARGIN = BINMASK (a binary image containing a
            %	 zero level set curve) then a signed distance function is generated
            %	 using BINMASK. BINMASK should have 1's inside and on the curve and
            %	 zeros elsewhere. If only the coordinates of the zero level set curve
            %	 are available, use custom function coord2mask to generate BINMASK
            %	 before using levelsetFunction. The level set function is computed
            % 	 directly (without iteration) using function bwdist.
            %
            %	 If TYPE = 'circular' and VARARGIN = M,N,X0,Y0,R then a circular
            %	 signed distance function array of size M-by-N, with center at
            %	 (X0,Y0) and radius R, is computed. The approach is illustrated in
            %	 Fig. 12.12 of DIPUM3E.
            %
            %   Copyright 2002-2020 Gatesmark
            %
            %   This function, and other functions in the DIPUM Toolbox, are based 
            %   on the theoretical and practical foundations established in the 
            %   book Digital Image Processing Using MATLAB, 3rd ed., Gatesmark 
            %   Press, 2020.
            %
            %   Book website: http://www.imageprocessingplace.com
            %   License: https://github.com/dipum/dipum-toolbox/blob/master/LICENSE.txt
            
            % COMPUTE THE LEVEL SET FUNCTION, DEPENDING ON SPECIFIED TYPE.
            switch type
	            case 'mask'
                  % Create the level set function as a signed distance function.    
                  binmask = varargin{1};
                  phi = double(bwdist(1 - binmask) - bwdist(binmask)...
                                                            - (binmask - 0.5));
	            case 'circular'
                  % A circle is automatically a signed distance function.
                  M = varargin{1};
                  N = varargin{2};
                  x0 = varargin{3};
                  y0 = varargin{4};
                  r = varargin{5};
                  [y,x] = meshgrid(1:N,1:M);
                  phi = sqrt((x - x0).^2 + (y - y0).^2) - r;
            end
        end
        
        function curveDisplay(x,y,varargin)
            narginchk(2,Inf)
            if nargin == 2
                % Default.
                plot(y,x,'.')
            else
                if ~isodd(length(varargin))
                    error('Wrong number of inputs.');
                end
            end
            plot(y,x,varargin{:});
        end
        
        
        function D = isodd(A)
            %ISODD Determines which elements of an array are odd numbers.
            %   D = ISODD(A) returns a logical array, D, of the same size as A,
            %   with 1s (TRUE) in the locations corresponding to odd numbers in
            %   A, and 0s (FALSE) elsewhere. 
            %
            %   Copyright 2002-2020 Gatesmark
            %
            %   This function, and other functions in the DIPUM Toolbox, are based 
            %   on the theoretical and practical foundations established in the 
            %   book Digital Image Processing Using MATLAB, 3rd ed., Gatesmark 
            %   Press, 2020.
            %
            %   Book website: http://www.imageprocessingplace.com
            %   License: https://github.com/dipum/dipum-toolbox/blob/master/LICENSE.txt
            
            D = 2*floor(A/2) ~= A;
        end
        
        function phinew = levelsetReset(phi,niter,delT)
            %levelsetReset Reinitializes a signed distance function.
            %   PHINEW = levelsetReset(PHI,NITER,delT) reinitializes level set
            %   function PHI so that its Euclidean norm is 1, as required of a
            %   signed distance function. This is done by evolving the equation
            %
            %               d/dt(phi) = sign(phi)(1 - ||grad(phi)||)
            %   
            %   At steady state, the term ||grad(phi)|| will equal 1 at all
            %   coordinates, within a stopping rule given by
            %
            %           sum(abs(phinew(:) - phi(:))) < delT*max(M,N)
            %
            %   
            %   NITER is the number of iterations--It defaults to 5. DELT is the
            %   time step used in iteration. Its default value is 0.5.
            %
            %   The method used here is based on the approach described in "A Level
            %   Set Approach for Computing Solutions to Incompressible Two-Phase
            %   Flow," by M. Sussman, Peter Smereka, and Stanley Osher, J. Comput.
            %   Phys., vol. 114, pp. 146-149, 1994.
            %
            %   Copyright 2002-2020 Gatesmark
            %
            %   This function, and other functions in the DIPUM Toolbox, are based 
            %   on the theoretical and practical foundations established in the 
            %   book Digital Image Processing Using MATLAB, 3rd ed., Gatesmark 
            %   Press, 2020.
            %
            %   Book website: http://www.imageprocessingplace.com
            %   License: https://github.com/dipum/dipum-toolbox/blob/master/LICENSE.txt
            
            % PRELIMINARIES.
            % Set defaults.
            if nargin == 2
               delT = 0.5;
            elseif nargin == 1
               delT = 0.5;
               niter = 5; 
            end
            % Size and sign of original function.
            [M,N] = size(phi);
            phi0 = phi;
            S = sign(phi0);
            
            % NORMALIZED GRADIENT.
            [phi0y,phi0x] = gradient(phi0);
            phi0 = phi0./(hypot(phi0x,phi0y) + eps);
            
            % ITERATE
            % To gain speed, the stopping rule is a modification of the one proposed
            % in the paper.
            for I = 1:niter
                phinew = phi - delT*S.*computeG(phi,phi0);
                if sum(abs(phinew(:) - phi(:))) < delT*max(M,N)
                    break
                end
                phi = phinew;
            end
        end
           
        %----------------------------------------------------------------------%
        function G = computeG(phi,phi0)
            % G is the gradient of all values of phi, but taking into account
            % where phi was positive, negative, or zero. This is necessary so
            % that movement during iteration will be in the correct direction.
            % The resulting G will be of size M-by-N.
            
            % Pad the input to handle derivative computations at border points. 
            % The padding adds one row/col all around.
            phi = padarray(phi,[1 1], 'replicate', 'both');
            
            % Size of padded array.
            [Mpad,Npad] = size(phi);
            
            % For computations of derivatives, the indices will then be 2:Mpad-1
            % and 2:Npad-1 on the padded result.
            
            % The following definitions are from pg 153 of the reference. We are
            % assuming a grid spacing of h = 1.
            i = 2:Mpad-1; % Row indices.
            j = 2:Npad-1; % Col indices.
            % a, b, c, and d are of size M-by-N.
            a = phi(i,j) - phi(i-1,j); % Backward difference in the x-direction.
            b = phi(i+1,j) - phi(i,j); % Forward difference in the x-direction.
            c = phi(i,j) - phi(i,j-1); % Backward difference in the y-direction.
            d = phi(i,j+1) - phi(i,j); % Forward difference in the y-direction.
            
            % Conditions on the derivatives to be used below.
            aplus = max(a,0); aminus = min(a,0);
            bplus = max(b,0); bminus = min(b,0);
            cplus = max(c,0); cminus = min(c,0);
            dplus = max(d,0); dminus = min(d,0);
            
            % The conditions on G (to be defined below) depend on the sign of
            % phi0.
            Ap = phi0 > 0;
            An = phi0 < 0;
            
            % G is the gradient of all values of phi, but taking into account
            % where phi was positive, negative, or zero. This is necessary so
            % that movement during iteration will be in the correct direction.
            % Note that G will be zero in locations where phi was zero.
            
            Gp = sqrt(max(aplus.^2, bminus.^2) + max(cplus.^2, dminus.^2)) - 1;
            Gn = sqrt(max(aminus.^2, bplus.^2) + max(cminus.^2, dplus.^2)) - 1;
            G = Gp.*Ap + Gn.*An; 
        
        end
        
        function phin = levelsetIterate(phi,F,delT)
            %levelsetIterate Iterative solution of level set equation.
            %   PHIN = levelsetIterate(PHI,F,delT) computes the iterative solution
            %   (see Eq. (12-55) in DIPUM3E) to the levelset equation. PHIN is the
            %   new value of the level set function PHI; F is the force term, and
            %   delT is the time increment, assumed to be in the range 0 < delT <=
            %   1. If delT is not included in the arguments, it defaults to delT =
            %   0.5*(1/max(F(:)). This function does one iteration, so generally it
            %   is called numerous times in a loop.
            %
            %   Copyright 2002-2020 Gatesmark
            %
            %   This function, and other functions in the DIPUM Toolbox, are based 
            %   on the theoretical and practical foundations established in the 
            %   book Digital Image Processing Using MATLAB, 3rd ed., Gatesmark 
            %   Press, 2020.
            %
            %   Book website: http://www.imageprocessingplace.com
            %   License: https://github.com/dipum/dipum-toolbox/blob/master/LICENSE.txt
            
            % SET DEFAULTS.
            if nargin == 2
                delT = 0.5*(1/max(F(:))); 
            
            else
            % Check to make sure deltT is valid.
                if delT > 1 || delT <=0
                    warning('delT should be in the range 0 < delT <= 1')
                end
            end
            
            % COMPUTE UPWIND DERIVATIVES (EQ. (12-58) IN DIPUM3E.
            [Dxplus,Dxminus,Dyplus,Dyminus] = upwindDerivatives(phi);
                
            % COMPUTE THE UPWIND GRADIENT MAGNITUDES (EQ. (12-57) IN DIPUM3E).
            [gMagPlus,gMagMinus] = upwindMagGrad(Dxplus,Dxminus,Dyplus,Dyminus);
            
            % UPDATE PHI (EQ. (12-55) IN DIPUM3E).
            phin = phi - delT*(max(F,0).*gMagPlus + min(F,0).*gMagMinus);
        
        end
        
        %----------------------------------------------------------------------%
        function [Dxplus,Dxminus,Dyplus,Dyminus] = upwindDerivatives(phi)
            %upwindDerivatives Computes first order upwind derivatives.
            %   [Dxplus,Dxminus,Dyplus,Dyminus] = upwindDerivatives(PHI)
            %   computes the upwind derivatives of level set function PHI using
            %   first-order approximations. The equations used are:
            %
            %   Dxplus  = phi(x + 1, y) - phi(x, y)
            %   Dxminus = phi(x, y) - phi(x - 1,y)
            %   Dyplus  = phi(x, y + 1) - phi(x,y)
            %   Dyminus = phi(x, y) - phi(x, y - 1)
            %
            %   The code is vectorized by using function circshift to perform
            %   the shifts required to implement the preceding expressions for
            %   all values of phi.
            
            % Use function circshift to displace coordinates for speedy
            % computation of the derivatives. Pad with a 1-pixel border of zeros
            % to prevent the derivatives from wrapping around as a result of the
            % circshift. Strip the border after the derivatives are computed.
            
            phi = padarray(phi, [1 1], 0 ,'both');
            
            Dxplus  = circshift(phi, 1, 1) - phi;
            Dxminus = phi - circshift(phi, -1, 1);
            Dyplus  = circshift(phi, 1, 2) - phi;
            Dyminus = phi - circshift(phi, -1, 2);
            
            % Strip out the border. Don't have to strip phi because it is not
            % passed back.
            Dxplus  = Dxplus(2:end-1, 2:end-1);
            Dxminus = Dxminus(2:end-1, 2:end-1);
            Dyplus  = Dyplus(2:end-1, 2:end-1);
            Dyminus = Dyminus(2:end-1, 2:end-1);
        
        end
        
        %----------------------------------------------------------------------%
        function [gMagPlus,gMagMinus] = upwindMagGrad(Dxplus,Dxminus,Dyplus,...
           Dyminus)
            %upwindMagGrad computes the upwind gradient magnitude.
            %   [gMagPlus,gMagMinus] = upwindMagGrad(Dxplus,Dxminus, Dyplus,
            %   Dyminus) computes the two components of the upwind normalized
            %   gradient of a level set function given the upwind derivatives
            %   Dxplus, Dxminus, Dyplus, and Dyminus of the function. These
            %   derivatives can be computed using function upWindDerivatives.
            
            gMagPlus  = sqrt((max(Dxminus,0).^2) + (min(Dxplus,0).^2) ...
                           + (max(Dyminus,0).^2) + (min(Dyplus,0).^2)); 
                  
            gMagMinus = sqrt((max(Dxplus,0).^2) + (min(Dxminus,0).^2) ...
                           + (max(Dyplus,0).^2) + (min(Dyminus,0).^2));
        end
                   
        %------------------------------------------------------------------------%
        
        function F = levelsetForce(type,paramcell,normcell)
            %LEVELSETFORCE Scalar force field for level-set segmentation.
            %   F = LEVELSETFORCE(TYPE,PARAMCELL,NORMCELL) computes a scalar force
            %	 field, F, of the type specified in TYPE:
            %
            %         TYPE                   Equation from Table 12.1
            %       'binary'                          (1)
            %       'gradient'                        (2)
            %       'geodesic'                        (3)
            %       'chanvese'                        (4) 
            %
            %	 In all cases, PARAMCELL is a cell array containing all the inputs
            %   required to implement the force specified in TYPE.
            %
            %	 NORMCELL (an optional input) is a cell array containing one of two
            %	 strings: NORMCELL = {str1, str2}. String srt1 can have one of two
            %	 values: 'Fn' or 'Fu'. The first string causes the force F to be
            %	 normalized by dividing it by its maximum absolute value. String 'Fu'
            %	 leaves F unchanged. String str2 is used only when option 'chanvese'
            %	 is used. It too can have one of two values: 'Cn' or 'Cu'. The first
            %	 form causes the curvature to be normalized by dividing it by its
            %	 maximum absolute value. The second form leaves the curvature
            %	 unchanged. If str2 is included in NORMCELL, then str1 must be
            %	 included also. The defaults if NORMCELL is not included in the
            %	 function call are 'Fu' and 'Cu'.
            %
            %	 Syntax forms for levelsetForce are as follows:
            %
            %   F = levelsetForce('binary',{f,a,b}, normcell) where f is a binary
            %	 image with values 0 and 1, implements the following function:
            %
            %                       F = a*f + b*(1 - f)
            %
            %	 This is Eq. (1) in Table 12.1. If force normalization is required,
            %	 let normcell = 'Fu'. For no normalization, do not include normcell
            %	 in the function call.
            %
            %	 F = levelsetForce('gradient',{f,p,lambda},normcell) implements the
            %	 following function:
            %   
            %           F = 1./(1 + lambda*(||gradient(f)||)^p)
            %
            %	 where ||arg|| is the vector norm of arg. This is Eq. (2) in Table
            %	 12.1. If force normalization is required, let normcell = {'Fn'}. For
            %	 no normalization, do not include normcell in the function call.
            %
            %   F = levelsetForce('geodesic',{phi,W,c},normcell) implements the
            %	 following function:
            %
            %        F = -div2D(W.*(phi./NORM(gradient(phi)))) - c*W
            %
            %	 where div2D computes the 2D divergence (see the Utility Functions
            %	 folder in the DIP4E Support Package). This is Eq. (3) in Table 12.1.
            %	 If force normalization is required, let normcell = {'Fn'}. For no
            %	 normalization, do not include normcell in the function call.
            %
            %	 F = levelsetForce('chanvese',{f,phi,mu,nu,lambda1,lambda2},normcell)
            %	 implements the following function:
            %
            %       F = -mu*div2D(phi) + nu + lamda1*((f - c1).^2) ... 
            %                                           - lambda2*((f - c2).^2)
            %
            %	 where DIPUM3E utility function div2D computes the 2D divergence
            %	 (curvature) of phi, as defined in Eq. (12-59) in DIPUM3E. If nu,
            %	 lambda1, and lambda2 are not included in the input argument, they
            %	 default to 0, 1, and 1, respectively. If force and/or curvature
            %	 normalization is desired, normcell must contain two character
            %	 strings: 'Fn' or 'Fu' for the force, and 'Cn' or 'Cu' for the
            %	 curvature. For example, to normalize both set normcell =
            %	 {'Fn','Cn'}. If neither normalization is needed, do not include
            %	 normcell in the function call.
            %
            %   Copyright 2002-2020 Gatesmark
            %
            %   This function, and other functions in the DIPUM Toolbox, are based 
            %   on the theoretical and practical foundations established in the 
            %   book Digital Image Processing Using MATLAB, 3rd ed., Gatesmark 
            %   Press, 2020.
            %
            %   Book website: http://www.imageprocessingplace.com
            %   License: https://github.com/dipum/dipum-toolbox/blob/master/LICENSE.txt
            
            % IMPLEMENT THE SELECTED FORCE.
            switch type
               case 'binary'
                  f = paramcell{1};
                  % Make sure the image is binary.
                  f = double(imbinarize(f));
                  a = paramcell{2};
                  b = paramcell{3};
                  F = a*f + b*(1 - f);
                  if nargin == 3 && isequal(normcell{1},'Fn') 
                     % Force normalization requested.
                     F = F./max(abs(F(:)));
                  end
	            case 'gradient'
                  f = paramcell{1};
                  p = paramcell{2};
                  lambda = paramcell{3};
                  [gy,gx] = gradient(f);
                  fnorm = sqrt(gx.^2 + gy.^2); 
                  F = 1./(1 + lambda*(fnorm.^p));
                  if nargin == 3 && isequal(normcell{1},'Fn') 
                     % Force normalization requested.
                     F = F./max(abs(F(:)));
                  end
	            case 'geodesic'
                  phi = paramcell{1};
                  c = paramcell{2};
                  W = paramcell{3};
                  % gradient is a MATLAB function.
                  [phiy, phix] = gradient(phi);
                  phinorm = sqrt(phix.^2 + phiy.^2);
                  phixN = phix./phinorm;
                  phiyN = phiy./phinorm;
                  % div2D is a utility function in the DIP4E Support Package.
                  F = -div2D(W.*phixN, W.*phiyN) - c*W;
                  if nargin == 3 && isequal(normcell{1},'Fn')
                     % Force normalization requested.
                     F = F./max(abs(F(:)));
                  end
	            case 'chanvese'
                  if numel(paramcell) == 3 % Default condition.
                     f = paramcell{1};
                     phi = paramcell{2};
                     mu = paramcell{3};
                     nu = 0;
                     lambda1 = 1;
                     lambda2 = 1;    
                  else
                     f = paramcell{1};
                     phi = paramcell{2};
                     mu = paramcell{3};
                     nu = paramcell{4};
                     lambda1 = paramcell{5};
                     lambda2 = paramcell{6};
                  end   
                  % Compute the average values of f at points inside and outside the
                  % contour. In the original algorithm, phi values on or inside the
                  % contour are defined as positive and values outside as negative.
                  % This is the opposite of the convention used in all other
                  % functions. We follow the algorithm in the book and then
                  % reconcile the difference at the end by changing the sign o the
                  % force.
                  idxIn = find(phi >= 0);
                  idxOut = find(phi < 0);
                  HS = levelsetHeaviside(phi,1);
                  c1 = sum(sum(f.*HS))/(length(idxIn) + eps); 
                  c2 = sum(sum(f.*(1 - HS)))/(length(idxOut) + eps);
                  % A simpler method that often works well is: 
                  %   c1 = mean2(f(phi >= 0));
                  %   c2 = mean2(f(phi < 0));
                  % Compute the curvature (check to see if curvature normalization
                  % is requested. Curvature normalization usually is important).
                  if nargin == 3 && numel(normcell) == 2 
                     mode = normcell{2};
                  else
                     % Normalization of curvature not requested.
                     mode = 'Cu'; 
                  end
                  V = levelsetCurvature(phi,mode);
                  % Compute the force.
                  F = mu*V + nu - lambda1*(f - c1).^2 + lambda2*(f - c2).^2;
                  % Change the sign of the force to correspond with our convention.
                  F = -F;
                  % Check to see if F should be normalized (normalization can slow
                  % down convergence).
                  if nargin == 3 && numel(normcell) == 2 && isequal(normcell{1},'Fn')
                     F = F/max(abs(F(:)));
                  end
            end
        end
        
        function [hs,imp] = levelsetHeaviside(phi,epsilon)
            %levelsetHeaviside 2D Heaviside and impulse for level set segmentation.
            %   HS = levelsetHeaviside(PHI,METHOD,EPSILON) computes a Heaviside
            %   function and its corresponding impulse (the derivative of the
            %   Heaviside function) for a 1-D or 2-D input function PHI, using the
            %   method suggested by Chan and Vese in the paper "Active Contours
            %   Without Edges," IEEE Trans. Image Processing, Vol. 10, no. 12, 2001,
            %   pp. 266-277. The equations implemented are:
            %
            %   Equation (12-68) in DIPUM3E:  
            %        HS = 0.5*(1 + (2/pi)*(1 + arctan(phi/epsilon)) 
            %   Equation (12-69):
            %        IMP = (1/pi)*epsilon./(epsilon^2 + phi.^2)
            %
            %   Copyright 2002-2020 Gatesmark
            %
            %   This function, and other functions in the DIPUM Toolbox, are based 
            %   on the theoretical and practical foundations established in the 
            %   book Digital Image Processing Using MATLAB, 3rd ed., Gatesmark 
            %   Press, 2020.
            %
            %   Book website: http://www.imageprocessingplace.com
            %   License: https://github.com/dipum/dipum-toolbox/blob/master/LICENSE.txt
            
            %   where epsilon is a constant that defaults to 1 when not included
            %   in the argument.
            
            % PRELIMINARIES
            % Default.
            if nargin == 1
                epsilon = 1;
            end
            % Make sure phi is floating point.
            phi = double(phi);
            
            % HEAVISIDE FUNCTION.
            hs = 0.5*(1 + (2/pi)*atan(phi/epsilon));
            
            % CORRESPONDING IMPULSE.
            imp = (1/pi)*epsilon./(epsilon^2 + phi.^2);
        
        end
        
        function K = levelsetCurvature(phi,mode)
            %levelsetCurvature Computes the curvature of a level set function.
            %   K = levelsetCurvature(PHI,MODE) computes the curvature, K, of level
            %   set function PHI. If MODE = 'Cu' the curvature is not normalized
            %   (this is the default). If MODE = 'Cn', the curvature is normalized
            %   by dividing it by its maximum value. This function is based on Eqs.
            %   (12-61) and (12-62) of DIPUM3E.
            %
            %   Copyright 2002-2020 Gatesmark
            %
            %   This function, and other functions in the DIPUM Toolbox, are based 
            %   on the theoretical and practical foundations established in the 
            %   book Digital Image Processing Using MATLAB, 3rd ed., Gatesmark 
            %   Press, 2020.
            %
            %   Book website: http://www.imageprocessingplace.com
            %   License: https://github.com/dipum/dipum-toolbox/blob/master/LICENSE.txt
            
            % PRELIMINARIES
            % Set default.
            if nargin == 1
                mode = 'Cu';
            end
            [M,N] = size(phi);
            phi = double(phi);
            
            % PAD WITH 1'S TO BE ABLE TO COMPUTE THE DERIVATIVE AT IMAGE BORDERS.
            phip = padarray(phi,[1 1],1,'both');
            
            % COMPUTE THE CENTRAL DIFFERENCES.
            phix = 0.5*(phip(3:end,2:N+1) - phip(1:M,2:N+1));
            phiy = 0.5*(phip(2:M+1,3:end) - phip(2:M+1,1:N));
            phixx = phip(3:end,2:N+1) + phip(1:M,2:N+1) - 2*phi;
            phiyy = phip(2:M+1,3:end) + phip(2:M+1,1:N) - 2*phi;
            phixy = 0.25.*(phip(3:end,3:end) - phip(1:M,3:end)...
                                        - phip(3:end,1:N) + phip(1:M,1:N));
            % COMPUTE THE CURVATURE.
            DEN = (phix.^2 + phiy.^2 + eps).^1.5;
            K = (phixx.*(phiy.^2) - 2*(phix.*phiy.*phixy) + phiyy.*(phix.^2))./DEN;
            
            % CHECK FOR NORMALIZATION.
            if isequal(mode,'Cn')
                K = K/max(abs(K(:)));
            end
             
            % COMPENSATE FOR BORDER EFFECTS BY SETTING THE FIRST AND LAST ROWS AND
            % COLUMNS EQUAL TO THEIR IMMEDIATE PREDECESSORS. 
            % CURVATURE 
            % Rows.
            K(:,1) = K(:,2);
            K(:,end) = K(:,end-1);
            % Columns.
            K(1,:) = K(2,:);
            K(end,:) = K(end-1,:);
        
        end