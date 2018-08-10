from factor import Factor 

def main():
    f1 = Factor(['Trav'], \
                [0.05, 0.95])
    
    f2 = Factor(['Fraud', 'Trav'], \
                [[0.01, 0.004], \
                 [0.99, 0.996]])

    f3 = Factor(['OC'], \
                [0.6, 0.4])
     
    f4 = Factor(['FP', 'Fraud', 'Trav'], \
                [[[0.9, 0.1], \
                  [0.9, 0.01]], \
                 [[0.1, 0.9], \
                 [0.1, 0.99]]])
      
    f5 = Factor(['IP', 'Fraud', 'OC'], \
                [[[0.02, 0.011], \
                  [0.01, 0.001]], \
                 [[0.98, 0.989], \
                  [0.99, 0.999]]])

    f6 = Factor(['CRP', 'OC'], \
                [[0.1, 0.001], \
                 [0.9, 0.999]])

    

    # Question b
    Factor.inference([f1, f2], ['Fraud'], ['Trav'], dict())
    Factor.inference([f1, f2, f3, f4, f5, f6], ['Fraud'], ['Trav', 'OC'], dict(FP = 't', IP = 'f', CRP = 't'))
    
    # Question c
    Factor.inference([f1, f2, f3, f4, f5, f6], ['Fraud'], ['OC'], dict(FP = 't', IP = 'f', CRP = 't', Trav = 't'))

    # Question d
    Factor.inference([f1, f2, f3, f4, f5, f6], ['Fraud'], ['Trav', 'FP', 'OC', 'CRP'], dict(IP = 't'))    
    Factor.inference([f1, f2, f3, f4, f5, f6], ['Fraud'], ['Trav', 'FP', 'OC'], dict(IP = 't', CRP = 't'))

main()