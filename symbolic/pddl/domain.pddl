(define (domain pick-place-and-stack)
  (:requirements :strips :negative-preconditions)

  ;;;;;;;;;; PREDICATES ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:predicates

    ;;; Domains ;;;
    (Graspable ?label); Name of a real object we can grasp
    (Base ?label); Name of a support object we CANNOT grasp
    (Waypoint ?pose) ; Model of any object we can go to in the world, real or not
    
    ;;; Objects ;;;
    (GraspObj ?label ?pose) ; The concept of a named object at a pose
    (PoseAbove ?pose ?label) ; The concept of a pose being supported by an object
  
    ;;; Object State ;;;
    (Free ?pose) ; This pose is free of objects, therefore we can place something here without collision
    (Supported ?labelUp ?labelDn) ; Is the "up" object on top of the "down" object?
    (Blocked ?label) ; This object cannot be lifted
    
    ;;; Robot State ;;;
    (HandEmpty)
    (Holding ?label)
    (AtPose ?pose)

  )

  ;;;;;;;;;; ACTIONS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (:action move_free
      :parameters (?poseBgn ?poseEnd)
      :precondition (and 
        ;; Robot State ;;
        (HandEmpty)
        (AtPose ?poseBgn)
      )
      :effect (and 
        ;; Robot State ;;
        (AtPose ?poseEnd)
        (not (AtPose ?poseBgn))
      )
  )
  
  (:action pick
    :parameters (?label ?pose ?prevSupport)
    :precondition (and
      ;; Domain ;;
      (Graspable ?label)
      (Waypoint ?pose)
      (Base ?prevSupport)
      ;; Object State ;;
      (GraspObj ?label ?pose)
      (Supported ?label ?prevSupport)
      (PoseAbove ?pose ?prevSupport)
      (not (Blocked ?label))
      ;; Robot State ;;
      (HandEmpty)
    )
    :effect (and
      ;; Robot State ;;
      (Holding ?label)
      (not (HandEmpty))
      ;; Object State ;;
      (not (Supported ?label ?prevSupport))
    )
  )
  
  (:action unstack
    :parameters (?label ?pose ?prevSupport)
    :precondition (and
      ;; Domain ;;
      (Graspable ?label)
      (Waypoint ?pose)
      (Graspable ?prevSupport)
      ;; Object State ;;
      (GraspObj ?label ?pose)
      (Supported ?label ?prevSupport)
      (PoseAbove ?pose ?prevSupport)
      (not (Blocked ?label))
      ;; Robot State ;;
      (HandEmpty)
    )
    :effect (and
      ;; Robot State ;;
      (Holding ?label)
      (not (HandEmpty))
      ;; Object State ;;
      (not (Supported ?label ?prevSupport))
      (not (Blocked ?prevSupport))
    )
  )
  
  (:action move_holding
      :parameters (?poseBgn ?poseEnd ?label)
      :precondition (and 
        ;; Robot State ;;
        (Holding ?label)
        (AtPose ?poseBgn)
        ;; Object State ;;
        (Free ?poseEnd)
        (GraspObj ?label ?poseBgn)
      )
      :effect (and 
        ;; Robot State ;;
        (AtPose ?poseEnd)
        (not (AtPose ?poseBgn))
        ;; Object State ;;
        (GraspObj ?label ?poseEnd)
        (not (GraspObj ?label ?poseBgn))
        (Free ?poseBgn)
        (not (Free ?poseEnd))
      )
  )
  
  (:action place
      :parameters (?label ?pose ?support)
      :precondition (and 
                      ;; Domain ;;
                      (Graspable ?label)
                      (Waypoint ?pose)
                      (Base ?support)
                      ;; Object State ;;
                      (GraspObj ?label ?pose) 
                      (PoseAbove ?pose ?support)
                      ;; Robot State ;;
                      (Holding ?label)
                      )
      :effect (and 
                ;; Robot State ;;
                (HandEmpty)
                (not (Holding ?label))
                (Supported ?label ?support)
              )
  )
  
  (:action stack
      :parameters (?labelUp ?poseUp ?labelDn)
      :precondition (and 
                      ;; Domain ;;
                      (Graspable ?labelUp)
                      (Waypoint ?poseUp)
                      (Graspable ?labelDn)
                      ;; Object State ;;
                      (GraspObj ?labelUp ?poseUp) 
                      (not (Blocked ?labelDn))
                      ;; Requirements ;;
                      (PoseAbove ?poseUp ?labelDn)
                      ;; Robot State ;;
                      (Holding ?labelUp)
                      )
      :effect (and 
                ;; Object State ;;
                (Supported ?labelUp ?labelDn)
                (Blocked ?labelDn)
                ;; Robot State ;;
                (HandEmpty)
                (not (Holding ?labelUp))
              )
  )
  
)