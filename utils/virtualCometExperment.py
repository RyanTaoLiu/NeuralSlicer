class NoOp:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            if len(args):
                print(args)
            if len(kwargs):
                print(kwargs)
        return method

# generate a class that do nothing
class virtualCometExperment(NoOp):
    def __init__(self):
        print('Using the local Logger')
