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
		(inhand ?a - movable)
		(on ?a - movable ?b - unmovable)
		(inworkspace ?a - movable)
	)
	(:action pick
		:parameters (?a - movable ?b - unmovable)
		:precondition (and
			(on ?a ?b)  ; TODO: Remove
			(forall (?b - movable)
				(and
					(not (inhand ?b))
					(not (on ?b ?a))
				)
			)
		)
		:effect (and
			(inhand ?a)
			(not (on ?a ?b))
		)
	)
	(:action place
		:parameters (?a - movable ?b - unmovable)
		:precondition (and
			(not (= ?a ?b))
			(inhand ?a)
		)
		:effect (and
			(not (inhand ?a))
			(on ?a ?b)
		)
	)
	(:action pull
		:parameters (?a - box ?b - tool)
		:precondition (and
			(not (= ?a ?b))
			(inhand ?b)
			(on ?a table)
		)
		:effect (and
			(inworkspace ?a)
		)
	)
)
