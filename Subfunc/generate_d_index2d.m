function d = generate_d_index2d(HIM2d, indexes , gt)

class_num = max(gt);

for k = 1 : class_num
    label = gt(indexes);
    temp_index = indexes(find(label == k));
     if isempty(temp_index) == 0
        d(k,:) = mean(HIM2d(temp_index,:));
     end
end