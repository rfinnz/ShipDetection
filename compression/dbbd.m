function [a] = dbbd(n2,m2)

a = zeros(1,1);
j = 1;

m = round(n2/m2);

%%%There should be m = n2/m2 number of 1s in ecah row. 
%%%Hence,for 25% CR, m would become a fraction, hence, that's not possible!
%%%So, we can't do 25% CR with DBBD! 

for i =1:m2
    a(i,j:j+m-1)=1;
    j = j+m;
end

end