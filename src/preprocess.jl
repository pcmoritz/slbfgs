function featurize(df)
  X = zeros(size(df, 1), 0)
  function onehot(column::DataArray)
    categories = unique(column)
    p = length(categories)
    n = size(column, 1)
    result = zeros(n, p)
    for (i, x) in enumerate(column)
      result[i,:] = eye(p)[:,findfirst(categories, x)]
    end
    return result
  end

  for feature in names(df)
    column = df[feature]
    if eltype(column) <: Number
      X = [X (column - mean(column))/std(column)]
    else
      X = [X onehot(column)]
    end
  end
  return X
end

function class(column)
  result = zeros(length(column))
  categories = unique(column)
  @assert length(categories) == 2
  for (i, x) in enumerate(column)
    if findfirst(categories, x) == 1
      result[i] = -1
    else
      @assert findfirst(categories, x) == 2
      result[i] = 1
    end
  end
  return result
end
