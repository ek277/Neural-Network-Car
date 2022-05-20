using System.Collections;
using System.Collections.Generic;
using UnityEngine;
// hasta aqui lo que hace es establecer las colecciones que predeterminadamente le permitiran hacer uso de objetos
// con el fin de hacer el codigo mas eficiente (como las librerias en python)
// incluso establece el entorno de unity donde se va a ejecutar el programa
[RequireComponent(typeof(NNet))]
// invoca el script donde se codifico la red neuronal para que la pueda usar este codigo
public class CarController : MonoBehaviour
{
    private Vector3 startPosition, startRotation; 
    private NNet network;

    // Define las variables, posicion inicial, giro inicial y la red neuronal

    [Range(-1f,1f)]    
    public float a,t;

    // las variables "de entrada" que se usaran en el modelo, a=aceleracion y t=tiempo
    // tambien establece los intervalos de actuacion, es decir, las cotas donde pueden moverse los valores (-1,1)

    public float timeSinceStart = 0f;

    // esto le permite al programa saber cuando ha estado divagando el auto por mucho tiempo y resetearse para evitar seguir linea recta infinita
    [Header("Fitness")]
    public float overallFitness;
    public float distanceMultipler = 1.4f;
    public float avgSpeedMultiplier = 0.2f;
    public float sensorMultiplier = 0.1f;
    
    // es la jerarquia de pesos, quiere decir que es mas importante identificar que auto llega mas lejos antes de una colision
    // si se corren varios autos a la vez y todos colisionan a la misma distancia, entonces entra el factor velocidad que
    // le permite al modelo elegira el auto que vaya mas rapido

    [Header("Network Options")]
    public int LAYERS = 1;
    public int NEURONS = 10;

    // las caracteristicas de la red, para este caso, 1 capa y 10 neuronas
    // usa estos valores por experiencia en este tipo de proyectos, el autor dice que hacerla mas compleja (mas capas y mas neuronas)
    // no necesariamente deriva en un auto mas inteligente considerando el computo necesario para ejecutar la tarea

    private Vector3 lastPosition;
    private float totalDistanceTravelled;
    private float avgSpeed;

    // esto le permite al modelo identificar la ultima posicion antes de la colision asi como la distancia recorrida y la velocidad media
    // estos valores calculan el fitness

    private float aSensor,bSensor,cSensor;

    // estos van a ser los inputs de nuestra red neuronal (posicion, distancia recorrida y velocidad media)
    private void Awake() {
        startPosition = transform.position;
        startRotation = transform.eulerAngles;
        network = GetComponent<NNet>();

        
    }// aqui se inicializan las variables principales y la red neuronal

    public void ResetWithNetwork (NNet net)
    {
        network = net;
        Reset();
    }

    // en esta parte pasaran todas las redes neuronales aleatorias que se generan en el script geneticmanager
    // para que el auto las ponga a prueba

    

    public void Reset() {

        timeSinceStart = 0f;
        totalDistanceTravelled = 0f;
        avgSpeed = 0f;
        lastPosition = startPosition;
        overallFitness = 0f;
        transform.position = startPosition;
        transform.eulerAngles = startRotation;
    }

    // esta es la funcion de reinicio, es decir, cada que el auto se reinicia (por accion de una colision)
    // todos los valores se reinician, de esta manera se evita que el auto no inicie "viciado" por el efecto del anterior

    private void OnCollisionEnter (Collision collision) {
        Death();
    }
    // si el auto colisiona, lo "mata", entonces se reinicia el ciclo y el auto pasa a la funcion "reset", se reinicia con
    // los valores arriba descritos

    private void FixedUpdate() {

        InputSensors();
        lastPosition = transform.position;


        (a, t) = network.RunNetwork(aSensor, bSensor, cSensor);

        // las variables aceleracion y tiempo seran controladas por la red neuronal misma que va tener parametros de entrenamiento
        // que son los sensores a, b y c


        MoveCar(a,t);

        timeSinceStart += Time.deltaTime;

        CalculateFitness();

       // el auto sera controlado por las variables aceleracion y tiempo


    }

    private void Death ()
    {
        GameObject.FindObjectOfType<GeneticManager>().Death(overallFitness, network);
    }

    // esta funcion establece con base en los calculos que hace en el geneticmanager para las redes neuronales cuando "morir"

    private void CalculateFitness() {

        totalDistanceTravelled += Vector3.Distance(transform.position,lastPosition);
        avgSpeed = totalDistanceTravelled/timeSinceStart;

       overallFitness = (totalDistanceTravelled*distanceMultipler)+(avgSpeed*avgSpeedMultiplier)+(((aSensor+bSensor+cSensor)/3)*sensorMultiplier);

       // en esta parte calcula cual es el mejor auto con base en los datos que ha recabado despues de someter pruebas de autos
       // considerando la funcion fitness al inicio, esto le permite generar una lista de "los autos que avanzaron mas
       // a una velocidad mas alta sin colisionar"

        if (timeSinceStart > 20 && overallFitness < 40) {
            Death();
        }

        // "mata" el proceso si despues de 20 segundos las condiciones no superan el valor 40, muchos valores (incluyendo este)
        //fueron determinados a base de prueba y error, el autor nos dice que no es necesario entenderlos todos, solo saber que funcionan

        if (overallFitness >= 1000) {
            Death();
        }

        // si las condiciones superan el valor de 1000, aniquila el proceso ya que ha cumplido con el proposito de este proyecto
        // recordemos que es un proyecto didactico por lo que de no hacerlo el auto continuara infinitamente recorriendo la pista

    }

    private void InputSensors() {

        Vector3 a = (transform.forward+transform.right);
        Vector3 b = (transform.forward);
        Vector3 c = (transform.forward-transform.right);

        //genera vectores a partir de los 3 sensores para determinar la posicion del auto (inputs), para definir los resultantes
        //puede ir hacia adelante, izquierda o derecha, estas ultimas 2 las define como sigue
        //izquierda = hacia adelante + izquierda
        // derecha = hacia adelante - izquierda
        //los vectores se construyen con la informacion que provee el codigo abajo

        Ray r = new Ray(transform.position,a);
        RaycastHit hit;

        if (Physics.Raycast(r, out hit)) {
            aSensor = hit.distance/20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

        // el auto registra colision a la derecha

        r.direction = b;

        if (Physics.Raycast(r, out hit)) {
            bSensor = hit.distance/20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

        //el auto registra colision adelante

        r.direction = c;

        if (Physics.Raycast(r, out hit)) {
            cSensor = hit.distance/20;
            Debug.DrawLine(r.origin, hit.point, Color.red);
        }

        //el auto registra colision a la izquierda

    }

    private Vector3 inp;
    public void MoveCar (float v, float h) {
        inp = Vector3.Lerp(Vector3.zero,new Vector3(0,0,v*11.4f),0.02f);
        inp = transform.TransformDirection(inp);
        transform.position += inp;

        transform.eulerAngles += new Vector3(0, (h*90)*0.02f,0);
    }

    // esta funcion lo que hace es establecer como avanza el auto, el autor explica que no es algo sencillo de explicar
    // no tiene una metodología para haber llegado a esta funcion mucho menos las constantes que ocupa

}
