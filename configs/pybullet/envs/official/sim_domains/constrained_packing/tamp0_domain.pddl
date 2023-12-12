(define (domain workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		unmovable - physobj
		movable - physobj
		tool - movable
		box - movable
	)
	(:constants table - unmovable)
	(:predicates
		(ingripper ?a - movable)
		(on ?a - movable ?b - physobj)
		(inworkspace ?a - movable)
	)
	(:action pick
		:parameters (?a - movable ?b - physobj)
		:precondition (and
			(on ?a ?b)
			(forall (?c - movable)
				(and
					(not (ingripper ?c))
					; (not (on ?c ?a))
				)
			)
		)
		:effect (and
			(ingripper ?a)
			(not (on ?a ?b))
		)
	)
	(:action place
		:parameters (?a - movable ?b - unmovable)
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
		:parameters (?a - box ?b - tool)
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
