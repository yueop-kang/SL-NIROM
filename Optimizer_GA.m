function [shape_opt,hist_opt]=Optimizer_GA(options,SearchScale,objectf)

N_D=options(1);
N_S=options(2);
N_P=options(3);
N_G=options(4);
p_t=options(5);
p_m=options(6);
N_cross=options(7);


% N_D dimension of search space
% Range range of search space

% N_S: length of string (presicion of decretization)a
% N_P: Population
% N_G: Generation

% p_t: tournament based wining rate
% p_m: mutation ratio
% p_c: crossover ratio
% N_mut: mutation limit
% N_cross: number of cutting

% N_tournamentSize: Tournament Size (should be divior of N_P)
N_tournamentSize=N_P/5;


% initialize string

Population=zeros(N_D,N_P,N_S);
Population_temp=zeros(N_D,N_P,N_S);
Population_inter=zeros(N_D,N_P,N_S);
Population_mutation=zeros(N_D,N_P,N_S);
Population_crossover=zeros(N_D,N_P,N_S);
Population_dec=zeros(N_D,N_P);
optima=zeros(1,N_G);

for i=1:N_D
    for j=1:N_P
        Population_temp(i,j,:)=rand(1,N_S);
    end
end


for i=1:N_D
    for j=1:N_P
        for k=1:N_S
            if Population_temp(i,j,k)< 0.5
                Population(i,j,k)=0;
            else
                Population(i,j,k)=1;
            end
        end
    end
end


% Iteration
min_optima=10^6;

for iter=1:N_G
    
    Population_dec=string2bin(N_S,N_D,N_P,Population,SearchScale);
    
    for i=1:N_P
        for j=1:N_D
            shape_input(j,i)=Population_dec(j,i);
        end
    end
    
    
    for i=1:N_P
        evaluation(i)=objectf(shape_input(:,i));
        EVAL(i)=evaluation(i);
    end
    
    
    tempi=0;
    
    for i=1:N_P
        Competitor_backnumber(i)=i;
    end
    
    for iter_tournament=1:fix(N_P/N_tournamentSize)
        
        Competitors=datasample(Competitor_backnumber,N_tournamentSize,'Replace',false);
        
        for i=1:N_tournamentSize
            evaluation_tournament(i)=evaluation(Competitors(i));
        end
        
        [eval_sort sorted_index]=sort(evaluation_tournament);
        
        
        p_f=zeros(1,N_tournamentSize);
        for i=1:N_tournamentSize
            if i==1
                p_f(sorted_index(i))=p_t;
            else if i==N_tournamentSize
                    temp=0;
                    for j=1:N_tournamentSize-1
                        temp=temp+p_t*(1-p_t)^(j-1);
                    end
                    p_f(sorted_index(i))=1-temp;
                else
                    p_f(sorted_index(i))=p_t*(1-p_t)^(i-1);
                end
            end
        end
        
        temp=0;
        
        for i=1:N_tournamentSize
            temp=temp+p_f(i);
            cumul_pf(i)=temp;
        end
        
        interpop=rand(1,N_tournamentSize);
        interN=zeros(1,N_tournamentSize);
        
        for i=1:N_tournamentSize
            temp=1;
            for j=1:N_tournamentSize
                if interpop(i)>cumul_pf(j)
                    temp=temp+1;
                end
            end
            interN(temp)=interN(temp)+1;
            
        end
        
        for i=1:N_tournamentSize
            if interN(i)>0
                for j=1:interN(i)
                    tempi=tempi+1;
                    for k=1:N_D
                        Population_inter(k,tempi,:)=Population(k,Competitors(i),:);  % verification is needed
                    end
                end
            end
        end
        
        if tempi==N_P
            break
        end
        
    end
    
    
    % Duplication is completed
    
    % Crossover
    
    Population_crossover=Population_inter;
    
    
    
    for i=1:N_P
        selection_temp(i)=i;
    end
    selection=datasample(selection_temp,N_P,'Replace',false);
    
    
    for cutn=1:N_cross
        for i=1:N_P/2
            cut=randi(N_S-1);
            for j=1:cut
                for k=1:N_D
                    temp=0;
                    temp=Population_crossover(k,selection(2*i-1),j);
                    Population_crossover(k,selection(2*i-1),j)=Population_crossover(k,selection(2*i),j);
                    Population_crossover(k,selection(2*i),j)=temp;
                end
            end
        end
    end
    
    for i=1:N_S
        sample_mut(i)=i;
    end
    
    
    
    % Mutation
    
    Population_mutation=Population_crossover;
    
    
    for i=1:N_P
        for k=1:N_D
            %             j=randi(N_S);
            for j=1:N_S
                flag_m=rand(1);
                if flag_m<p_m
                    
                    if Population_mutation(k,i,j)==0
                        Population_mutation(k,i,j)=1;
                    else
                        Population_mutation(k,i,j)=0;
                    end
                    
                end
            end
        end
    end
    
    % update
    optima(iter)=min(EVAL);
    Population=Population_mutation;
    
    
    for i=1:N_P
        if EVAL(i)<min_optima
            min_optima=EVAL(i);
            optima_index=i;
            shape=Population_dec(:,optima_index)';
        end
    end
    
    if iter==N_G
        min_optima;
    end
    
end

shape_opt=shape';
hist_opt=optima;
end

function Population_dec=string2bin(N_S,N_D,N_P,Population,SearchScale)

sequence=zeros(N_D,N_P);

for i=1:N_D
    for j=1:N_P
        sequence(i,j)=bin2dec(Population(i,j,:));
    end
end

if SearchScale == "log"
    
    L=(3-(-2))*1/2^N_S;
    bin=[-2:L:3];
    
    
    for i=1:N_D
        for j=1:N_P
            Population_dec(i,j)=10^(rand(1)*L+bin(sequence(i,j)+1));
        end
    end
end

if SearchScale == "linear"
    
    L=(1-(0))*1/2^N_S;
    bin=[0:L:1];

    for i=1:N_D
        for j=1:N_P
            Population_dec(i,j)=rand(1)*L+bin(sequence(i,j)+1);
        end
    end
end

if ~(SearchScale=="log" ||SearchScale=="linear")
    fprintf('GA option "SearchScale" should be log or linear ')
end



end


function y=bin2dec(string)

N_S=length(string);

for i=1:N_S
    bin(i)=2^(i-1);
end

temp=0;
for i=1:N_S
    temp=temp+bin(i)*string(i);
end

y=temp;

end

