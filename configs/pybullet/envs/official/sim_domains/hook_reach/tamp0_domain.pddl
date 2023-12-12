(define (domain workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		movable - physobj
		unmovable - physobj
		tool - movable
		box - movable
	)
	(:constants table - physobj)
	(:predicates
		(ingripper ?a - movable)
		(on ?a - movable ?b - physobj)
		(inworkspace ?a - movable)
	)
	(:action pick
		:parameters (?a - movable ?b - physobj)
		:precondition (and
			(on ?a ?b)  ; TODO: Remove
			(forall (?b - movable)
				(and
					(not (ingripper ?b))
					(not (on ?b ?a))
				)
			)
		)
		:effect (and
			(ingripper ?a)
			(forall (?b - physobj) (not (on ?a ?b)))
		)
	)
	(:action place
		:parameters (?a - movable ?b - physobj)
		:precondition (and
			(not (= ?a ?b))
			(ingripper ?a)
		)
		:effect (and
			(not (ingripper ?a))
			(on ?a ?b)
		)
	)
	(:action pull
		:parameters (?a - movable ?b - movable)
		:precondition (and
			(not (= ?a ?b))
			(ingripper ?b)
			(on ?a table)
		)
		:effect (and
			(inworkspace ?a)
		)
	)
)
