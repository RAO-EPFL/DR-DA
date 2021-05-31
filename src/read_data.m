function domain_org = read_data(data_name)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequential Domain Adaptation by Synthesizing Distributionally Robust
% Experts
% ICML 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function reads data from csv files.
% INPUT:
% dataname : name of the dataset ("california_housing", "life_expectancy",
% "birth_USA", "houses", "uber")
% OUTPUT:
% domain_org : source&target domain data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

    if data_name == "california_housing"
        data = readtable('./data/california_housing.csv');
        data = rmmissing(data);
        shifting_prop = string(data.ocean_proximity);
        ind1 = find(shifting_prop == 'NEAR BAY');
        ind2 = find(shifting_prop == 'INLAND');
        ind3 = find(shifting_prop == '<1H OCEAN');

        y = normalize(data.median_house_value);
        data.median_house_value = [];
        data.ocean_proximity = [];
        x = table2array(data);
        x1 = normalize(x(ind1, :), 1); y1 = normalize(y(ind1, :));
        x2 = normalize(x(ind2, :), 1); y2 = normalize(y(ind2, :));
        x3 = normalize(x(ind3, :), 1); y3 = normalize(y(ind3, :));
        x_source = x3;
        y_source = y3;
        x_target = x2;
        y_target = y2;
    elseif data_name == "life_expectancy"
        data = readtable('./data/LifeExpectancyDataset.csv');
        data = rmmissing(data);
        data.Country = [];
        data.Year = [];

        shifting_prop = string(data.Status);
        ind1 = find(shifting_prop == 'Developing');
        ind2 = find(shifting_prop == 'Developed');
        data.Status = [];
        y = normalize(data.LifeExpectancy);
        data.LifeExpectancy = [];
        x = table2array(data);

        x1 = normalize(x(ind1, :), 1); y1 = normalize(y(ind1, :));
        x2 = normalize(x(ind2, :), 1); y2 = normalize(y(ind2, :));
        x_source = x1; y_source = y1;
        x_target = x2; y_target = y2;
    elseif data_name == "insurance"
        data = readtable('./data/insurance.csv');
        data = rmmissing(data);
        [N d] = size(data);
        y = table2array(data(:, d));
        bias = string(data.region);
        indices_se = find(bias == 'southeast');
        indices_sw = find(bias == 'southwest');
        indices_ne = find(bias == 'northeast');
        indices_nw = find(bias == 'northwest');
        x = zeros(N, d-1);
        x(:, 1) = table2array(data(:, 1));
        x(find(string(data.sex) == 'female'), 2) = 1;
        x(:, 3) = table2array(data(:, 3));
        x(:, 4) = table2array(data(:, 4));

        x(find(string(data.smoker) == 'yes'), 5) = 1;

        x(find(string(data.region) == 'southeast'), 6) = 1;
        x(find(string(data.region) == 'southwest'), 6) = 2;
        x(find(string(data.region) == 'northeast'), 6) = 3;
        x(find(string(data.region) == 'northwest'), 6) = 4;
        [N, d] = size(x);

        x1 = normalize(x(find(x(:, 6) == 1 | x(:, 6) == 3), 1:5), 1); 
        y1 = normalize(y(find(x(:, 6) == 1 | x(:, 6) == 3)));
        
        x2 = normalize(x(find(x(:, 6) == 2 | x(:, 6) == 4), 1:5), 1);
        y2 = normalize(y(find(x(:, 6) == 2 | x(:, 6) == 4)));
        
        x_source = x1 ;
        y_source = y1;

        x_target = x2;
        y_target = y2;
        
    elseif data_name == "houses"
        data_train = readtable('./data/houses/train.csv');
        data_train = rmmissing(data_train);
        x = zeros(1121, 13);
        data_train.Id = [];
        data_train.MSSubClass = [];
        data_train.MSZoning = [];
        x(:, 1) = normalize(data_train.LotFrontage);
        x(:, 2) = normalize(data_train.LotArea);
        % data_train.Street = [];
        % data_train.Alley = [];
        % data_train.LotShape = [];
        % data_train.LandContour = [];
        % data_train.Utilities = [];
        % data_train.LotConfig = [];
        % data_train.LandSlope = [];
        % data_train.Neighborhood = [];
        % data_train.Condition1 = [];
        % data_train.Condition2 = [];
        % data_train.BldgType = [];
        % data_train.HouseStyle = [];
        x(:, 3) = normalize(data_train.OverallQual);
        x(:, 4) = normalize(data_train.OverallCond);
        x(:, 5) = normalize(data_train.TotalBsmtSF);
        x(:, 6) = normalize(data_train.x1stFlrSF);
        x(:, 7) = normalize(data_train.x2ndFlrSF);
        x(:, 8) = normalize(data_train.GrLivArea);
        x(:, 9) = normalize(data_train.BsmtFullBath);
        x(:, 10) = normalize(data_train.BsmtHalfBath);
        x(:, 11) = normalize(data_train.BedroomAbvGr);
        % x(:, 12) = normalize(data_train.KitchenAbvGr);
        x(:, 12) = normalize(data_train.TotRmsAbvGrd);
        x(:, 13) = normalize(data_train.GarageArea);

        shifting_prop = data_train.YearBuilt;
        y = normalize(data_train.SalePrice);

        ind1 = find(shifting_prop >= 1950 & shifting_prop < 2000);
        ind2 = find(shifting_prop >= 2000 & shifting_prop <= 2010);
%         ind3 = find(shifting_prop > 1990 & shifting_prop <= 2010);
        % ind4 = find(shifting_prop >= 2010 & shifting_prop < 1990);
        % ind5 = find(shifting_prop >= 1990 & shifting_prop < 2000);
        % ind6 = find(shifting_prop >= 2000 & shifting_prop < 2016);

        x1 = x(ind1, :); y1 = y(ind1, :);
        x2 = x(ind2, :); y2 = y(ind2, :);
%         x3 = x(ind3, :); y3 = y(ind3, :);
        % x4 = x(ind4, :); y4 = y(ind4, :);
        % x5 = x(ind5, :); y5 = y(ind5, :);
        % x6 = x(ind6, :); y6 = y(ind6, :);


        %% Specify Source and Target
        x_source = x1;
        y_source = y1;

        x_target = x2;
        y_target = y2;
        
    elseif data_name == "birth_USA"
        data = readtable('./data/kaggle/birth_data/USbirths_2018.csv');
        data = rmmissing(data);
        %%
        y = data.DBWT;
        data.DBWT = [];
        %%
        shift_ = string(data.SEX);
        %%
        data.SEX = [];
        %%
        shift_(find(shift_ == "F")) = 1;
        shift_(find(shift_ == "M")) = 0;
        %%
        shift_ = shift_(1:10000);
        shift_ = double(shift_);
        %% 
        % xa = string(data.LD_INDL);
        % xa(find(xa == "N")) = 0; 
        % xa(find(xa == "Y")) = 1; 
        % xa = double(xa(1 : 10000));
        % data.LD_INDL = [];
        % 
        % xb = string(data.RF_CESAR);
        % xb(find(xb == "N")) = 0; 
        % xb(find(xb == "Y")) = 1; 
        % xb = double(xb(1 : 10000));
        % data.RF_CESAR = [];
        %%
        data.LD_INDL = [];
        data.RF_CESAR = [];
        x = table2array(data);
        x = x(1:10000, :);
        % clear data
        %%
        y = y(1:10000);
        % x = [x, xa, xb];
        % clear xa xb
        %%
        x1 = normalize(x(find(shift_ == 0), :), 1);
        x2 = normalize(x(find(shift_ == 1), :), 1);
        y1 = normalize(y(find(shift_ == 0), :), 1);
        y2 = normalize(y(find(shift_ == 1), :), 1);
        %%
        x_source = x1;
        y_source = y1;
        x_target = x2;
        y_target = y2;
    elseif data_name == "uber"
        data = readtable('./data/kaggle/uber/rideshare_kaggle.csv');
        data = rmmissing(data);

        data.timezone = [];
        data.source = [];
        data.product_id = [];

        data.surge_multiplier = [];
        data.latitude = [];
        data.longitude = [];
        data.moonPhase = [];
        data.destination = [];
        data.id = [];
        data.timestamp = [];
    
        data.long_summary = [];

        
        data.windGustTime = [];
        
        data.datetime = [];
        

        indices = find(data.month == 12 & data.day <= 15);
        
        data = data(indices, :);

        
        shift = data.cab_type;
        data.cab_type = [];
        y = data.price;
        data.price = [];
        
        temp = datetime(data.temperatureHighTime, 'ConvertFrom', 'posixtime');
        data.temperatureHighTime = hour(temp);
        
        temp = datetime(data.temperatureLowTime, 'ConvertFrom', 'posixtime');
        data.temperatureLowTime = hour(temp);
        
        data.name = [];
        
        [s1, s2] = size(data);
        x_icon = zeros(s1, 1);
        x_icon(string(data.icon) == " cloudy ") = 1;
        x_icon(string(data.icon) == " rain ") = 2;
        x_icon(string(data.icon) == " fog ") = 3;
        x_icon(string(data.icon) == " clear-night ") = 4;
        x_icon(string(data.icon) == " clear-day ") = 5;
        x_icon(string(data.icon) == " partly-cloudy-day ") = 6;
        x_icon(string(data.icon) == " partly-cloudy-night ") = 7;
        data.icon = [];
        
        data.month = [];
        data.day = [];
        
        [s1, s2] = size(data);
        x_short_summary = zeros(s1, 1);
        x_short_summary(string(data.short_summary) == " Light Rain ") = 1;
        x_short_summary(string(data.short_summary) == " Foggy ") = 2;
        x_short_summary(string(data.short_summary) == " Clear ") = 3;
        x_short_summary(string(data.short_summary) == " Mostly Cloudy ") = 4;
        x_short_summary(string(data.short_summary) == " Partly Cloudy ") = 5;
        x_short_summary(string(data.short_summary) == " Light Rain ") = 6;
        x_short_summary(string(data.short_summary) == " Overcast ") = 7;
        x_short_summary(string(data.short_summary) == " Possible Drizzle ") = 8;
        
        data.short_summary = [];
        
        temp = datetime(data.apparentTemperatureHighTime, 'ConvertFrom', 'posixtime');
        data.apparentTemperatureHighTime = hour(temp);
        
        temp = datetime(data.apparentTemperatureLowTime, 'ConvertFrom', 'posixtime');
        data.apparentTemperatureLowTime = hour(temp);
        
        temp = datetime(data.apparentTemperatureMaxTime, 'ConvertFrom', 'posixtime');
        data.apparentTemperatureMaxTime = hour(temp);
        
        temp = datetime(data.temperatureMinTime, 'ConvertFrom', 'posixtime');
        data.temperatureMinTime = hour(temp);

        
        temp = datetime(data.temperatureMaxTime, 'ConvertFrom', 'posixtime');
        data.temperatureMaxTime = hour(temp);
        
        temp = datetime(data.sunriseTime, 'ConvertFrom', 'posixtime');
        data.sunriseTime = hour(temp);
        
        temp = datetime(data.sunsetTime, 'ConvertFrom', 'posixtime');
        data.sunsetTime = minute(temp);
        
        temp = datetime(data.uvIndexTime, 'ConvertFrom', 'posixtime');
        data.uvIndexTime = hour(temp);
        
        shift = string(shift);
        x1_inds = find(shift == "Uber");
        x1_inds = x1_inds(1:5000);
        x2_inds = find(shift == "Lyft");
        x2_inds = x2_inds(1:5000);
        y1 = normalize(y(x1_inds));
        y2 = normalize(y(x2_inds));
        
        x = table2array(data);
        x1 = normalize(x(x1_inds, :), 1);
        x2 = normalize(x(x2_inds, :), 1);
        
        x_source = x1;
        y_source = y1;
        x_target = x2;
        y_target = y2;
    end

    
    domain_org = {};
    domain_org.source = {};
    domain_org.target = {};
    domain_org.source.x = x_source;
    domain_org.source.y = y_source;
    domain_org.target.x = x_target;
    domain_org.target.y = y_target;
    [~, d] = size(x_source);
    domain_org.dimension = d;
    
end