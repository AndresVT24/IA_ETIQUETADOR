__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    
    # Funciones de análisis cualitativo y cuantitativo
    
    # cualitativo
    
    def Retrieval_by_color(images, color_labels, query_colors):
        """
        Cerca imatges que contenen unes etiquetes de color.

        Args:
            images: llista d'imatges.
            color_labels: llista d'etiquetes de color predites per KMeans.
            query_colors: string o llista de strings amb els colors que volem buscar.

        Returns:
            Llista d'imatges que contenen els colors demanats.
        """
        pass
    
    def Retrieval_by_shape(images, shape_labels, query_shape):
        """
        Cerca imatges que tenen una etiqueta de forma concreta.

        Args:
            images: llista d'imatges.
            shape_labels: llista d'etiquetes de forma predites per KNN.
            query_shape: string amb la forma que volem buscar.

        Returns:
            Llista d'imatges que tenen la forma demanada.
        """
        pass
    
    def Retrieval_combined(images, color_labels, shape_labels, query_colors, query_shape):
        """
        Cerca imatges que compleixen alhora una etiqueta de color i una etiqueta de forma.

        Args:
            images: llista d'imatges.
            color_labels: llista d'etiquetes de color predites per KMeans.
            shape_labels: llista d'etiquetes de forma predites per KNN.
            query_colors: string o llista de strings amb els colors buscats.
            query_shape: string amb la forma buscada.

        Returns:
            Llista d'imatges que compleixen la cerca combinada.
        """
        pass
    
    # cuantitativas
    
    def Kmean_statistics(images, k_values, options=None):
        """
        Calcula estadístiques d'execució de KMeans per diferents valors de K.

        Args:
            images: llista d'imatges sobre les quals executar KMeans.
            k_values: llista de valors de K a provar.
            options: diccionari opcional amb paràmetres de KMeans.

        Returns:
            Diccionari amb estadístiques com WCD, temps d'execució i nombre d'iteracions.
        """
        pass

    def Get_shape_accuracy(predicted_shapes, gt_shapes):
        """
        Calcula l'accuracy de les etiquetes de forma obtingudes amb KNN.

        Args:
            predicted_shapes: llista d'etiquetes de forma predites.
            gt_shapes: llista d'etiquetes de forma del Ground Truth.

        Returns:
            Accuracy de forma en percentatge o valor entre 0 i 1.
        """
        pass
    
    def Get_color_accuracy(predicted_colors, gt_colors, partial_match=True):
        """
        Calcula l'accuracy de les etiquetes de color obtingudes amb KMeans.

        Args:
            predicted_colors: llista d'etiquetes de color predites per KMeans.
            gt_colors: llista d'etiquetes de color del Ground Truth.
            partial_match: si True, permet comptar coincidències parcials.

        Returns:
            Accuracy de color.
        """
        pass