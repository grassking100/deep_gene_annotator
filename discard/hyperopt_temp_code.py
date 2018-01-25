def Hyperopt_setting(args):
    convolution_number=args['convolution_number']
    lstm_number=args['lstm_number']
    convolution_settings=Convolution_layers_settings()
    if convolution_number>=1:
        c1_num=args['c1_num']
        c1_size=args['c1_size']
        convolution_settings.add_layer(c1_num,c1_size)
    if convolution_number>=2:
        c2_num=args['c2_num']
        c2_size=args['c2_size']  
        convolution_settings.add_layer(c2_num,c2_size)
    if convolution_number>=3:
        c3_num=args['c3_num']
        c3_size=args['c3_size']  
        convolution_settings.add_layer(c3_num,c3_size)
    return (convolution_settings.get_settings(),lstm_number)
def objective(args):
    convolution_settings,lstm_number=Hyperopt_setting(args)
    print(convolution_settings,lstm_number)
    model=Exon_intron_finder_factory(convolution_settings,lstm_number)
    evaluator=Model_evaluator()
    evaluator.add_traning_data(x_train,y_train)
    evaluator.add_validation_data(x_test,y_test)
    evaluator.evaluate(model,30,30,True,0)
    (accuracy_arr,val_accuracy_arr)=evaluate_model(model,30,x_train,y_train,x_test,y_test,30)
    return {
        'loss': -evaluator.get_last_accuracy(),
        'val_loss': -evaluator.get_last_validation_accuracy(),
        'max_loss': -evaluator.get_max_accuracy(),
        'max_val_loss': -evaluator.get_max_validation_accuracy(),
        'status': STATUS_OK,
        }
best = fmin(objective,
    space=hp.pchoice('p_options,',[
        (.1,{
            'convolution_number':0,
            'lstm_number':hp.randint('lstm_number_0',30)+1
        }),
        (.2,{
            'convolution_number':1,
            'c1_num':hp.randint('c1_num_1',118)+10,
            'c1_size':hp.randint('c1_size_1',248)+10,
            'lstm_number':hp.randint('lstm_number_1',30)+1
        }),
        (.5,{
            'convolution_number':2,
            'c1_num':hp.randint('c1_num_2',118)+10,
            'c1_size':hp.randint('c1_size_2',248)+10,
            'c2_num':hp.randint('c2_num_2',118)+10,
            'c2_size':hp.randint('c2_size_2',248)+10,
            'lstm_number':hp.randint('lstm_number_2',30)+1
        }),
        (.2,{
            'convolution_number':3,
            'c1_num':hp.randint('c1_num_3',118)+10,
            'c1_size':hp.randint('c1_size_3',248)+10,
            'c2_num':hp.randint('c2_num_3',118)+10,
            'c2_size':hp.randint('c2_size_3',248)+10,
            'c3_num':hp.randint('c3_num_3',118)+10,
            'c3_size':hp.randint('c3_size_3',248)+10,
            'lstm_number':hp.randint('lstm_number_3', 30)+1
        })
    ]),
    algo=tpe.suggest,
    max_evals=50,
    trials=trials)