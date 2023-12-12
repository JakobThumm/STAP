(define (domain workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		unmovable - physobj
		movable - physobj
		tool - movable
		box - movable
		receptacle - unmovable
	)
	(:constants table - unmovable)
	(:predicates
		(ingripper ?a - movable)
		(on ?a - movable ?b - unmovable)
		(inworkspace ?a - physobj)
		(beyondworkspace ?a - physobj)
		(under ?a - movable ?b - receptacle)
		(aligned ?a - physobj)
		(poslimit ?a - physobj)
	)
	(:action pick
		:parameters (?a - physobj)
		:precondition (and
            (or
                ; 1) (state) Movable ingripper, 
                ; 1a)       (action) Pick physobj. 
                (exists (?b - movable) (ingripper ?b))
                
                ; 2) (state) Nothing ingripper, 
                ; 2a)       (action) Pick unmovable.
                (and 
                    (forall (?b - movable) (not (ingripper ?b)))
                    (exists (?b - unmovable) (= ?a ?b))
                )
            )
        )
		:effect (and )
	)
	(:action place
		:parameters (?a - physobj ?b - physobj)
		:precondition (and
			(not (= ?a ?b))
            (or
                ; 1) (state) Movable ingripper.
                ; 1a)       (action) ingripper(?a), place on another movable.
                (and (ingripper ?a) (exists (?c - movable) (= ?b ?c)))
                ; 1b)       (action) (not (ingripper ?a)), place on anything.
                (exists (?c - movable) (and (ingripper ?c) (not (= ?a ?c))))

                ; 2) (state) Nothing ingripper. 
                ; 2a)       (action) Place anything on anything.
                (forall (?c - movable) (not (ingripper ?c)))
            )
		)
		:effect (and )
    )
	(:action pull
		:parameters (?a - physobj ?b - physobj)
		:precondition (and
			(not (= ?a ?b))
            (or 
                ; 1) (state) Movable ingripper.
                ; 1a)       (state) ingripper(?b), b is a tool.
                (and (ingripper ?b) (exists (?c - tool) (= ?b ?c))
                    (or
                ; 1ax)              (action) Pull movable on rack with tool.
                        (and 
                            (exists (?c - movable) (= ?a ?c))
                            (exists (?c - receptacle) (on ?a ?c))
                        )
                ; 1ay)              (action) Pull unmovable with tool.
                        (exists (?c - unmovable) (= ?a ?c))
                    )
                )
                
                ; 1b)       (state) ingripper(?b), b is a box.
                ; 1bx)              (action) Pull anything with box.
                (and (ingripper ?b) (exists (?c - box) (= ?b ?c)))

                ; 1c)       (state) (not (ingripper ?b)), 
                ; 1cx)              (action) Pull anything with anything.
                (not (ingripper ?b))
                
                ; 2) (state) Nothing ingripper.
                ; 2a)       (action) Pull anything with anything.
                (forall (?c - movable) (not (ingripper ?c)))
            )
		)
		:effect (and )
	)
    (:action push
        :parameters (?a - physobj ?b - physobj ?c - receptacle)
        :precondition (and
            (not (= ?a ?b))
            (not (= ?a ?c))
            (not (= ?b ?c))
            (or
                ; 1) (state) Movable ingripper
                ; 1a)       (state) ingripper(?b), b is a tool
                (and (ingripper ?b) (exists (?d - tool) (= ?b ?d))
                    (or
                ; 1ax)              (action) Push movable on rack with tool.
                        (and
                            (exists (?d - movable) (= ?a ?d))
                            (exists (?d - receptacle) (on ?a ?d))
                        )
                ; 1ay)              (action) Push ummovable with tool.
                        (exists (?d - unmovable) (= ?a ?d))
                    )
                )
                ; 1b)       (state) ingripper(?b), b is a box
                ; 1bx)              (action) Push anything with box.
                (and (ingripper ?b) (exists (?d - box) (= ?b ?d)))
                
                ; 1c)       (state) (not (ingripper ?b))
                ; 1cx)              (action) Push anything with anything.
                (not (ingripper ?b))

                ; 2) (state) Nothing ingripper.
                ; 2a)       (action) Push anything with anything.
                (forall (?d - movable) (not (ingripper ?d)))
            )
        )
        :effect (and )
    )
)