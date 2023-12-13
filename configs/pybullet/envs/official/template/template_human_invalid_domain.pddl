(define (domain workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		unmovable - physobj
		movable - physobj
		tool - movable
		box - movable
		receptacle - unmovable
        actor - physobj
	)
	(:constants table - unmovable)
	(:predicates
		(ingripper ?a - movable)
        (inhand ?a - movable)
		(on ?a - movable ?b - unmovable)
        (accepting ?a - actor)
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
                (inhand ?a)
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
                (inhand ?a)
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
    (:action static_handover
		:parameters (?a - physobj ?b - physobj)
		:precondition (or
			(not (ingripper ?a))
			(not (accepting ?b))
            (and 
                (forall (?c - movable) (not (ingripper ?c)))
                (exists (?c - unmovable) (= ?a ?c))
            )
		)
		:effect (and )
	)
)