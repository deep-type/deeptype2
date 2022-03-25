def make_callable(obj):
    if isinstance(obj, str):
        name, function_kwargs = obj, {}
    else:
        def parse_function(function, args={}):
            args = args.copy()
            return function, args
        name, function_kwargs = parse_function(**obj)
    module_path, function_name = name.rsplit(':', 1)
    result = getattr(__import__(module_path, fromlist=(function_name,)), function_name)
    if len(function_kwargs) > 0:
        def result_wrapper(*args, **kwargs):
            actual_kwargs = function_kwargs.copy()
            actual_kwargs.update(kwargs)
            return result(*args, **actual_kwargs)
        return result_wrapper
    else:
        return result
