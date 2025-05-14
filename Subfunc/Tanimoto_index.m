function [TI] = Tanimoto_index(map1, map2)
[x, y, z] = size(map1);
for i = 1:z
    idx1_1 = find(map1(:,:,i)==1);
    idx1_2 = find(map2(:,:,i)==1);
    [c, ia, ib] = intersect(idx1_1, idx1_2);
    A = length(c);
    B = length(union(idx1_1, idx1_2));
    TI_all(i) = A/B;
end
TI = mean(TI_all);