import math
from register_automata.raa import RAA_Transition
from register_automata.raa import RAA
from z3 import Real  


def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))


def create_hamming_raa():
    trans1 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") < Real("curr2"),  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 1],  
        move = 3
    )

    trans2 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") == Real("curr2"),  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 0],  
        move = 3
    )

    trans3 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") > Real("curr2"),  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 1],  
        move = 3
    )

    raa_ham = RAA(
        states = [0],
        initial_state = 0,
        final_states = [0],
        transitions= [trans1, trans2, trans3],
        register_vars = [])

    return raa_ham

def create_l1_raa():
    trans1 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") <= Real("curr2"),  
        assignment = {}, 
        end_state = 0,
        cost_function = [-1, 1, 0],  
        move = 3
    )

    trans2 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") > Real("curr2"),  
        assignment = {}, 
        end_state = 0,
        cost_function = [1, -1, 0],  
        move = 3
    )

    raa_l1 = RAA(
        states = [0],
        initial_state = 0,
        final_states = [0],
        transitions= [trans1, trans2],
        register_vars = [])

    return raa_l1


def create_edit_raa():
    trans1 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") != Real("curr2"),  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 1],  
        move = 3
    )

    trans2 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") == Real("curr2"),  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 0],  
        move = 3
    )

    trans3 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = True,  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 2],  
        move = 3
    )

    trans4 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = True,  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 1],  
        move = 1
    )

    trans5 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = True,  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 1],  
        move = 2
    )

    raa_edit = RAA(
        states = [0],
        initial_state = 0,
        final_states = [0],
        transitions= [trans1, trans2, trans3, trans4, trans5],
        register_vars = [])

    return raa_edit


def create_dtw_raa():
    """Return the RAA for Dynamic Time Warping with L1 local cost.

    A DTW alignment may advance the first input, the second input, or both.
    Each transition therefore has one of the three corresponding head moves;
    its local cost is ``abs(curr1 - curr2)``.  The three guards express that
    absolute value as an affine RAA cost, just as ``create_l1_raa`` does.
    """
    curr1 = Real("curr1")
    curr2 = Real("curr2")
    cost_cases = (
        (curr1 < curr2, [-1, 1, 0]),
        (curr1 == curr2, [0, 0, 0]),
        (curr1 > curr2, [1, -1, 0]),
    )
    transitions = [
        RAA_Transition(
            cv1=curr1,
            cv2=curr2,
            start_state=0,
            guard=guard,
            assignment={},
            end_state=0,
            cost_function=cost,
            move=move,
        )
        for guard, cost in cost_cases
        for move in (1, 2, 3)
    ]

    return RAA(
        states=[0],
        initial_state=0,
        final_states=[0],
        transitions=transitions,
        register_vars=[],
    )



# def create_dtw_raa():
#     raise NotImplementedError("create_dtw_raa has been moved to raa.py. Import from there.")
#         end_state = 0,
#         cost_function = [-1, 1, 0],  
#         move = 2
#     )
#     trans3 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") < Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [-1, 1, 0],  
#         move = 3
#     )


#     trans4 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") > Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [1, -1, 0],  
#         move = 1
#     )

#     trans5 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") > Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [1, -1, 0],  
#         move = 2
#     )

#     trans5 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") > Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [1, -1, 0],  
#         move = 3
#     )

#     trans6 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") == Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [0, 0, 0],  
#         move = 1
#     )

#     trans7 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") == Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [0, 0, 0],  
#         move = 2
#     )

#     trans8 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") == Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [0, 0, 0],  
#         move = 3
#     )
 

#     raa_dtw = RAA(
#         states = [0],
#         initial_state = 0,
#         final_states = [0],
#         transitions= [trans1, trans2, trans3, trans4, trans5, trans6, trans7, trans8],
#         register_vars = [])

#     return raa_dtw


def create_last_letter_raa():
    trans1 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") == Real("curr2"),  
        assignment = {}, 
        end_state = 0,
        cost_function = [0, 0, 0],  
        move = 3
    )

    trans2 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") >= Real("curr2"),  
        assignment = {}, 
        end_state = 1,
        cost_function = [1, -1, 0],  
        move = 3
    )

    trans3 = RAA_Transition(
        cv1 = Real("curr1"),
        cv2 = Real("curr2"),
        start_state= 0,
        guard = Real("curr1") < Real("curr2"),  
        assignment = {}, 
        end_state = 1,
        cost_function = [-1, 1, 0],  
        move = 3
    )

    raa_last_letter = RAA(
        states = [0, 1],
        initial_state = 0,
        final_states = [1],
        transitions= [trans1, trans2, trans3],
        register_vars = [])
    
    return raa_last_letter


# def create_dtw_raa():
#     trans1 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") == Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [0, 0, 0],  
#         move = 3
#     )

#     trans2 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") == Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [0, 0, 0],  
#         move = 2
#     )

#     trans3 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") == Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [0, 0, 0],  
#         move = 1
#     )
    

#     trans4 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") < Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [-1, 1, 0],  
#         move = 3
#     )

#     trans5 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") < Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [-1, 1, 0],  
#         move = 2
#     )


#     trans6 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") < Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [-1, 1, 0],  
#         move = 1
#     )
 
#     trans7 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") > Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [1, -1, 0],  
#         move = 3
#     )

#     trans8 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") > Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [1, -1, 0],  
#         move = 2
#     )
#     trans9 = RAA_Transition(
#         cv1 = Real("curr1"),
#         cv2 = Real("curr2"),
#         start_state= 0,
#         guard = Real("curr1") > Real("curr2"),  
#         assignment = {}, 
#         end_state = 0,
#         cost_function = [1, -1, 0],  
#         move = 1
#     )   

    

#     raa_dtw = RAA(
#         states = [0],
#         initial_state = 0,
#         final_states = [0],
#         transitions= [trans1,trans2,trans3,
#                       trans4,trans5,trans6,
#                       trans7,trans8,trans9,
#                       ],
#         register_vars = [])
#     return raa_dtw



# if __name__ == "__main__":
    # raa  = create_dtw_raa()
    # print(raa.run_all_paths([1,2,3,7,9], [1,2,3,7,10]))
