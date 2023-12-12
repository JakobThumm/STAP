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
        (inhand ?a - movable)
		(on ?a - movable ?b - unmovable)
		(inworkspace ?a - physobj)
		(beyondworkspace ?a - physobj)
		(under ?a - movable ?b - receptacle)
		(aligned ?a - physobj)
		(poslimit ?a - physobj)
	)
	(:action pick
		:parameters (?a - movable)
		:precondition (and
			(exists (?b - unmovable) (on ?a ?b))
			(forall (?b - movable) (not (ingripper ?b)))
            (not (inhand ?a))
		)
		:effect (and
			(ingripper ?a)
			(forall (?b - unmovable) (not (on ?a ?b)))
		)
	)
    (:action static_handover
		:parameters (?a - movable ?b - receptacle)
		:precondition (and
			(ingripper ?a)
			(forall (?c - movable) (not (inhand ?c)))
		)
		:effect (and
			(not (ingripper ?a))
            (inhand ?a)
			(forall (?b - unmovable) (not (on ?a ?b)))
		)
	)
	(:action place
		:parameters (?a - movable ?b - unmovable)
		:precondition (and
			(not (= ?a ?b))
            (not (inhand ?a))
			(ingripper ?a)
		)
		:effect (and
			(not (ingripper ?a))
			(on ?a ?b)
		)
	)
)