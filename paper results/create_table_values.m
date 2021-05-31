clc
data_set = "houses";
load([data_set + ".mat"])
t = [5, 10, 50, 100];
emp1 = cumloss_track_emp(:, :, 1);
emp1 = mean(emp1, 1);

CL_RWS = mean(cumloss_track_DITL, 1);
CL_CCL = mean(cumloss_track_emp(:, :, 1), 1);
CL_CCSL = mean(cumloss_track_emp(:, :, 2), 1);
CL_CCTL = mean(cumloss_track_emp(:, :, 3), 1);
CL_CCTE = mean(cumloss_track_emp(:, :, 4), 1);
CL_CCSE = mean(cumloss_track_emp(:, :, 5), 1);

CL_IRKL = mean(cumloss_track_IRKL, 1); 
CL_IRWASS = mean(cumloss_track_IRWASS, 1);
CL_SIKL = mean(cumloss_track_SIKL, 1);
CL_SIWASS = mean(cumloss_track_SIWASS, 1);

CL_OPT_T = mean(cumloss_track_lse_alltarget, 1);
CL_OPT_TS = mean(cumloss_track_lse_alltarget_source, 1);
names = ["IR-KL", "IR-WASS", "SI-KL", "SI-WASS", "CC-L", "CC-TL", "CC-SL", "CC-TE", "CC-SE", "RWS", "LSE-T", "LSE-T&S", ...
     ];
cnt = 1;
CLS_vals = zeros(4, 12);
for t = [5, 10 ,50 , 100]
    CLS = [CL_IRKL(t); CL_IRWASS(t); ...
            CL_SIKL(t); CL_SIWASS(t); CL_CCL(t); CL_CCTL(t); CL_CCSL(t); ...
            CL_CCTE(t); CL_CCSE(t); CL_RWS(t);
            CL_OPT_T(t); CL_OPT_TS(t) ];

    norm_min = min(CLS);
    norm_max = max(CLS);
    CLS_vals(cnt, :) = CLS / min(CLS);
    cnt = cnt + 1;
end

T = table([names; CLS_vals]);
