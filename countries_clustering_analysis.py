#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无监督聚类分析：基于地理坐标的国家聚类
目标：将全球国家按照七大洲进行无监督聚类分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
import warnings
warnings.filterwarnings('ignore')

# Set English locale for matplotlib
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """Load and preprocess the country data"""
    # Load data
    df = pd.read_csv('countries_geocodes.csv')
    print(f"Original data shape: {df.shape}")
    
    # Remove rows with missing values
    df = df.dropna()
    print(f"Data shape after removing missing values: {df.shape}")
    
    # Create comprehensive continent mapping for 6 continents (excluding Antarctica)
    continent_mapping = {
        # Asia
        'Afghanistan': 'Asia', 'Armenia': 'Asia', 'Azerbaijan': 'Asia', 'Bahrain': 'Asia',
        'Bangladesh': 'Asia', 'Bhutan': 'Asia', 'Brunei': 'Asia', 'Cambodia': 'Asia',
        'China': 'Asia', 'Cyprus': 'Asia', 'Georgia': 'Asia', 'India': 'Asia',
        'Indonesia': 'Asia', 'Iran': 'Asia', 'Iraq': 'Asia', 'Israel': 'Asia',
        'Japan': 'Asia', 'Jordan': 'Asia', 'Kazakhstan': 'Asia', 'Kuwait': 'Asia',
        'Kyrgyzstan': 'Asia', 'Laos': 'Asia', 'Lebanon': 'Asia', 'Malaysia': 'Asia',
        'Maldives': 'Asia', 'Mongolia': 'Asia', 'Myanmar': 'Asia', 'Nepal': 'Asia',
        'North Korea': 'Asia', 'Oman': 'Asia', 'Pakistan': 'Asia', 'Palestine': 'Asia',
        'Philippines': 'Asia', 'Qatar': 'Asia', 'Russia': 'Asia', 'Saudi Arabia': 'Asia',
        'Singapore': 'Asia', 'South Korea': 'Asia', 'Sri Lanka': 'Asia', 'Syria': 'Asia',
        'Taiwan': 'Asia', 'Tajikistan': 'Asia', 'Thailand': 'Asia', 'Timor-Leste': 'Asia',
        'Turkey': 'Asia', 'Turkmenistan': 'Asia', 'United Arab Emirates': 'Asia',
        'Uzbekistan': 'Asia', 'Vietnam': 'Asia', 'Yemen': 'Asia', 'Hong Kong': 'Asia',
        'Macau': 'Asia',
        
        # Europe  
        'Albania': 'Europe', 'Andorra': 'Europe', 'Austria': 'Europe', 'Belarus': 'Europe',
        'Belgium': 'Europe', 'Bosnia': 'Europe', 'Bulgaria': 'Europe', 'Croatia': 'Europe',
        'Czech Republic': 'Europe', 'Denmark': 'Europe', 'Estonia': 'Europe', 'Finland': 'Europe',
        'France': 'Europe', 'Germany': 'Europe', 'Greece': 'Europe', 'Hungary': 'Europe',
        'Iceland': 'Europe', 'Ireland': 'Europe', 'Italy': 'Europe', 'Latvia': 'Europe',
        'Liechtenstein': 'Europe', 'Lithuania': 'Europe', 'Luxembourg': 'Europe', 'Macedonia': 'Europe',
        'Malta': 'Europe', 'Moldova': 'Europe', 'Monaco': 'Europe', 'Montenegro': 'Europe',
        'Netherlands': 'Europe', 'Norway': 'Europe', 'Poland': 'Europe', 'Portugal': 'Europe',
        'Romania': 'Europe', 'San Marino': 'Europe', 'Serbia': 'Europe', 'Slovakia': 'Europe',
        'Slovenia': 'Europe', 'Spain': 'Europe', 'Sweden': 'Europe', 'Switzerland': 'Europe',
        'UK': 'Europe', 'Ukraine': 'Europe', 'Vatican City': 'Europe',
        'Faroe Islands': 'Europe', 'Gibraltar': 'Europe', 'Guernsey': 'Europe', 'Isle of Man': 'Europe', 
        'Jersey': 'Europe', 'Svalbard & Jan Mayen': 'Europe', 'Aland Islands': 'Europe',
        
        # Africa
        'Algeria': 'Africa', 'Angola': 'Africa', 'Benin': 'Africa', 'Botswana': 'Africa',
        'Burkina Faso': 'Africa', 'Burundi': 'Africa', 'Cameroon': 'Africa', 'Cape Verde': 'Africa',
        'Central African Republic': 'Africa', 'Chad': 'Africa', 'Comoros': 'Africa',
        'Congo - Brazzaville': 'Africa', 'Congo - Kinshasa': 'Africa', 'Cote dIvoire': 'Africa',
        'Djibouti': 'Africa', 'Egypt': 'Africa', 'Equatorial Guinea': 'Africa', 'Eritrea': 'Africa',
        'Ethiopia': 'Africa', 'Gabon': 'Africa', 'Gambia': 'Africa', 'Ghana': 'Africa',
        'Guinea': 'Africa', 'Guinea-Bissau': 'Africa', 'Kenya': 'Africa', 'Lesotho': 'Africa',
        'Liberia': 'Africa', 'Libya': 'Africa', 'Madagascar': 'Africa', 'Malawi': 'Africa',
        'Mali': 'Africa', 'Mauritania': 'Africa', 'Mauritius': 'Africa', 'Morocco': 'Africa',
        'Mozambique': 'Africa', 'Namibia': 'Africa', 'Niger': 'Africa', 'Nigeria': 'Africa',
        'Rwanda': 'Africa', 'Sao Tome & Principe': 'Africa', 'Senegal': 'Africa', 'Seychelles': 'Africa',
        'Sierra Leone': 'Africa', 'Somalia': 'Africa', 'South Africa': 'Africa', 'South Sudan': 'Africa',
        'Sudan': 'Africa', 'Swaziland': 'Africa', 'Tanzania': 'Africa', 'Togo': 'Africa',
        'Tunisia': 'Africa', 'Uganda': 'Africa', 'Zambia': 'Africa', 'Zimbabwe': 'Africa',
        'Mayotte': 'Africa', 'Reunion': 'Africa', 'St. Helena': 'Africa', 'Western Sahara': 'Africa',
        
        # North America
        'Antigua & Barbuda': 'North America', 'Bahamas': 'North America', 'Barbados': 'North America',
        'Belize': 'North America', 'Canada': 'North America', 'Costa Rica': 'North America',
        'Cuba': 'North America', 'Dominica': 'North America', 'Dominican Republic': 'North America',
        'El Salvador': 'North America', 'Grenada': 'North America', 'Guatemala': 'North America',
        'Haiti': 'North America', 'Honduras': 'North America', 'Jamaica': 'North America',
        'Mexico': 'North America', 'Nicaragua': 'North America', 'Panama': 'North America',
        'St. Kitts & Nevis': 'North America', 'St. Lucia': 'North America', 'St. Vincent & Grenadines': 'North America',
        'Trinidad & Tobago': 'North America', 'US': 'North America',
        'Anguilla': 'North America', 'Aruba': 'North America', 'Bermuda': 'North America', 
        'British Virgin Islands': 'North America', 'Cayman Islands': 'North America', 'Curacao': 'North America', 
        'Greenland': 'North America', 'Guadeloupe': 'North America', 'Martinique': 'North America',
        'Montserrat': 'North America', 'Puerto Rico': 'North America', 'Sint Maarten': 'North America', 
        'St. Barthelemy': 'North America', 'St. Martin': 'North America', 'St. Pierre & Miquelon': 'North America',
        'Turks & Caicos Islands': 'North America', 'U.S. Virgin Islands': 'North America',
        'Caribbean Netherlands': 'North America',
        
        # South America
        'Argentina': 'South America', 'Bolivia': 'South America', 'Brazil': 'South America',
        'Chile': 'South America', 'Colombia': 'South America', 'Ecuador': 'South America',
        'French Guiana': 'South America', 'Guyana': 'South America', 'Paraguay': 'South America',
        'Peru': 'South America', 'Suriname': 'South America', 'Uruguay': 'South America',
        'Venezuela': 'South America', 'Falkland Islands': 'South America',
        'South Georgia & South Sandwich Islands': 'South America',
        
        # Oceania
        'Australia': 'Oceania', 'Fiji': 'Oceania', 'Kiribati': 'Oceania', 'Marshall Islands': 'Oceania',
        'Micronesia': 'Oceania', 'Nauru': 'Oceania', 'New Zealand': 'Oceania', 'Palau': 'Oceania',
        'Papua New Guinea': 'Oceania', 'Samoa': 'Oceania', 'Solomon Islands': 'Oceania',
        'Tonga': 'Oceania', 'Tuvalu': 'Oceania', 'Vanuatu': 'Oceania',
        'American Samoa': 'Oceania', 'Cook Islands': 'Oceania', 'French Polynesia': 'Oceania', 
        'Guam': 'Oceania', 'New Caledonia': 'Oceania', 'Niue': 'Oceania', 'Norfolk Island': 'Oceania',
        'Northern Mariana Islands': 'Oceania', 'Pitcairn Islands': 'Oceania', 'Tokelau': 'Oceania',
        'Wallis & Futuna': 'Oceania', 'Christmas Island': 'Oceania', 'Cocos (Keeling) Islands': 'Oceania',
        'Heard & McDonald Islands': 'Oceania', 'U.S. Outlying Islands': 'Oceania',
        
        # Antarctica (will be removed as outlier) - ONLY Antarctica itself
        'Antarctica': 'Antarctica'
    }
    
    # Add continent information
    df['continent'] = df['country_name'].map(continent_mapping)
    
    # Check for unmapped countries
    unmapped = df[df['continent'].isnull()]
    if len(unmapped) > 0:
        print("Warning: Unmapped countries found:")
        print(unmapped['country_name'].tolist())
        # Remove unmapped countries
        df = df[df['continent'].notnull()]
    
    # Identify Antarctica (outlier to be removed) - ONLY Antarctica itself
    antarctica_mask = df['country_name'] == 'Antarctica'
    antarctica_data = df[antarctica_mask].copy()
    
    # Remove Antarctica from main dataset (outlier removal)
    main_data = df[~antarctica_mask].copy()
    
    print(f"\nDataset sizes:")
    print(f"Original (with Antarctica): {len(df)}")
    print(f"Antarctica countries: {len(antarctica_data)}")
    print(f"Main dataset (no Antarctica): {len(main_data)}")
    
    print(f"\nContinent distribution:")
    print(main_data['continent'].value_counts())
    
    return main_data, antarctica_data

def optimal_clusters_analysis(data):
    """Determine optimal number of clusters using elbow method"""
    features = ['latitude', 'longitude']
    X = data[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of clusters
    cluster_range = range(2, 15)
    inertias = []
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve only
    plt.figure(figsize=(10, 6))
    
    plt.plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters', fontsize=14, fontweight='bold')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=14, fontweight='bold')
    plt.title('Elbow Method for Optimal k', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=6, color='red', linestyle='--', linewidth=2, label='k=6 (chosen)')
    plt.legend(fontsize=12)
    
    # Add annotation for the chosen k
    plt.annotate('Optimal k=6', xy=(6, inertias[4]), xytext=(8, inertias[4] + 200),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scaler

def perform_clustering(data, n_clusters=6):
    """Perform clustering using K-means and Hierarchical clustering"""
    features = ['latitude', 'longitude']
    X = data[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize clustering algorithms
    algorithms = {
        'K-means': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'Hierarchical': AgglomerativeClustering(n_clusters=n_clusters)
    }
    
    results = {}
    
    # Create continent to number mapping for evaluation
    continents = sorted(data['continent'].unique())
    continent_to_num = {cont: i for i, cont in enumerate(continents)}
    true_labels = [continent_to_num[cont] for cont in data['continent']]
    
    for name, algorithm in algorithms.items():
        # Fit the algorithm
        labels = algorithm.fit_predict(X_scaled)
        
        # Calculate metrics
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else -1
        
        results[name] = {
            'labels': labels,
            'ari': ari,
            'nmi': nmi,
            'silhouette': silhouette,
            'n_clusters': len(np.unique(labels))
        }
        
        print(f"{name} Clustering Results:")
        print(f"  Number of clusters: {len(np.unique(labels))}")
        print(f"  ARI: {ari:.3f}")
        print(f"  NMI: {nmi:.3f}")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print()
    
    return results, scaler, true_labels, continents

def create_world_map_with_continents(data, antarctica_data, clustering_results, method_name):
    """Create world map with continent choropleth and clustering results using built-in continent mapping"""
    
    # Create the figure
    fig = go.Figure()
    
    # Prepare clustering data
    plot_data = data.copy()
    plot_data['cluster'] = clustering_results[method_name]['labels']
    
    # Define cluster colors (vibrant, distinct colors)
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create base choropleth map with continent coloring using built-in country features
    # Define continent colors
    continent_colors = {
        'Asia': '#FFE6CC',           # Light Orange
        'Europe': '#E6D7FF',         # Light Purple  
        'Africa': '#FFE0D4',         # Light Orange-Red
        'North America': '#D4F4DD',  # Light Green
        'South America': '#D4E6FF',  # Light Blue
        'Oceania': '#FFE0F0',        # Light Pink
        'Antarctica': '#FFD4D4'      # Light Red
    }
    
    # Create country-level mapping for choropleth
    country_continent_map = {}
    continent_color_map = {}
    
    for _, row in data.iterrows():
        country = row['country_name']
        continent = row['continent']
        
        # Map common country name variations to ISO codes for choropleth
        country_iso_map = {
            'US': 'USA', 'UK': 'GBR', 'Russia': 'RUS', 'Congo - Kinshasa': 'COD',
            'Congo - Brazzaville': 'COG', 'Cote dIvoire': 'CIV', 'South Korea': 'KOR',
            'North Korea': 'PRK', 'Bosnia': 'BIH', 'Macedonia': 'MKD'
        }
        
        iso_country = country_iso_map.get(country, country)
        country_continent_map[iso_country] = continent
        continent_color_map[iso_country] = continent_colors.get(continent, '#F0F0F0')
    
    # Add Antarctica data if exists
    if len(antarctica_data) > 0:
        country_continent_map['Antarctica'] = 'Antarctica'
        continent_color_map['Antarctica'] = continent_colors['Antarctica']
    
    # Create choropleth for continent background
    fig.add_trace(go.Choropleth(
        locations=list(country_continent_map.keys()),
        z=[list(continent_colors.keys()).index(cont) for cont in country_continent_map.values()],
        locationmode='country names',
        colorscale=[[i/6, color] for i, color in enumerate(continent_colors.values())],
        showscale=False,
        hoverinfo='skip',
        marker_line_color='white',
        marker_line_width=0.5
    ))
    
    # Add clustered countries by cluster (on top of continent backgrounds)
    for cluster in sorted(plot_data['cluster'].unique()):
        cluster_data = plot_data[plot_data['cluster'] == cluster]
        fig.add_trace(go.Scattergeo(
            lon=cluster_data['longitude'],
            lat=cluster_data['latitude'],
            text=cluster_data['country_name'] + '<br>Continent: ' + cluster_data['continent'],
            mode='markers',
            marker=dict(
                size=8,
                color=cluster_colors[cluster % len(cluster_colors)],
                line=dict(width=2, color='white'),
                opacity=1.0
            ),
            name=f'Cluster {cluster}',
            hovertemplate='<b>%{text}</b><br>Longitude: %{lon}<br>Latitude: %{lat}<br>Cluster: ' + str(cluster) + '<extra></extra>'
        ))
    
    # Add Antarctica separately (outlier)
    if len(antarctica_data) > 0:
        fig.add_trace(go.Scattergeo(
            lon=antarctica_data['longitude'],
            lat=antarctica_data['latitude'],
            text=antarctica_data['country_name'],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(width=3, color='black'),
                opacity=1.0
            ),
            name='Antarctica (Outlier)',
            hovertemplate='<b>%{text}</b><br>Longitude: %{lon}<br>Latitude: %{lat}<br>Status: Outlier (not clustered)<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'World Map - {method_name} Clustering Results<br><sub>6 Continents Analysis (Antarctica excluded from clustering)</sub>',
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(150, 150, 150)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True,
            countrycolor='rgb(180, 180, 180)',
            countrywidth=0.8,
        ),
        title_x=0.5,
        width=1400,
        height=800,
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Save files
    filename = f'world_map_{method_name}.html'
    img_filename = f'world_map_{method_name}.png'
    
    fig.write_html(filename)
    fig.write_image(img_filename, width=1400, height=800, scale=2.5)
    fig.show()
    
    print(f"World map saved as {filename} and {img_filename}")

def plot_confusion_matrix(data, clustering_results, method_name, true_labels, continents):
    """Create confusion matrix between true continents and clustering results"""
    predicted_labels = clustering_results[method_name]['labels']
    
    # Create confusion matrix using continent names as labels
    continent_names = continents
    cluster_names = [f'Cluster {i}' for i in range(len(np.unique(predicted_labels)))]
    
    # Build confusion matrix manually
    cm = np.zeros((len(continent_names), len(cluster_names)))
    
    for i, continent in enumerate(continent_names):
        continent_mask = data['continent'] == continent
        continent_clusters = predicted_labels[continent_mask]
        
        for cluster_id in continent_clusters:
            cm[i, cluster_id] += 1
    
    # Calculate accuracy for each continent
    continent_accuracies = {}
    for i, continent in enumerate(continent_names):
        if cm.sum(axis=1)[i] > 0:
            best_cluster = np.argmax(cm[i, :])
            accuracy = cm[i, best_cluster] / cm.sum(axis=1)[i] * 100
            continent_accuracies[continent] = accuracy
        else:
            continent_accuracies[continent] = 0
    
    # Plot confusion matrix with better styling
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=cluster_names,
                yticklabels=continent_names,
                cbar_kws={'label': 'Number of Countries'})
    plt.title(f'Confusion Matrix - {method_name} Clustering\n(6 Continents, k=6)', fontsize=14, pad=20)
    plt.xlabel('Predicted Cluster', fontsize=12)
    plt.ylabel('True Continent', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{method_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return continent_accuracies

def plot_accuracy_comparison(accuracies_dict):
    """Plot accuracy comparison between K-means and Hierarchical clustering"""
    # Prepare data for plotting
    continents = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']
    methods = ['K-means', 'Hierarchical']
    
    # Create data matrix
    data_matrix = []
    for continent in continents:
        row = []
        for method in methods:
            accuracy = accuracies_dict.get(method, {}).get(continent, 0)
            row.append(accuracy)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(continents))
    width = 0.35
    
    # Light blue and orange color scheme
    colors = ['#87CEEB', '#FFA07A']  # Light blue and light salmon (orange-ish)
    
    for i, method in enumerate(methods):
        bars = ax.bar(x + i * width - width/2, data_matrix[:, i], width, 
                     label=method, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Continent', fontsize=14, fontweight='bold')
    ax.set_ylabel('Clustering Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Clustering Accuracy Comparison by Continent\n(6 Continents, k=6, Excluding Antarctica)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(continents, rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 115)  # Increased y-axis limit to give more space
    
    # Add average accuracy text with original horizontal positioning
    for i, method in enumerate(methods):
        avg_acc = np.mean(data_matrix[:, i])
        ax.text(0.02 + i*0.2, 0.95, f'{method} Avg: {avg_acc:.1f}%', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('clustering_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def hierarchical_parameter_selection(data):
    """Hierarchical clustering parameter selection: linkage methods comparison"""
    features = ['latitude', 'longitude']
    X = data[features]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different linkage methods
    linkage_methods = ['ward', 'complete', 'average', 'single']
    n_clusters_range = range(2, 15)
    
    # Compute distance matrix for linkage methods that need it
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, cophenet
    
    distance_matrix = pdist(X_scaled, metric='euclidean')
    
    # Calculate cophenetic correlation coefficients for different linkage methods
    cophenetic_scores = {}
    dendrograms = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hierarchical Clustering: Linkage Methods Comparison\n(Euclidean Distance)', 
                 fontsize=16, fontweight='bold')
    
    for i, method in enumerate(linkage_methods):
        # Calculate linkage matrix
        Z = linkage(distance_matrix, method=method)
        
        # Calculate cophenetic correlation coefficient
        coph_dists, _ = cophenet(Z, distance_matrix)
        cophenetic_scores[method] = coph_dists
        
        # Plot dendrogram
        ax = axes[i//2, i%2]
        
        from scipy.cluster.hierarchy import dendrogram
        dendrogram(Z, ax=ax, truncate_mode='level', p=5, 
                  color_threshold=0.7*max(Z[:,2]))
        
        ax.set_title(f'{method.capitalize()} Linkage\nCophenetic Corr: {coph_dists:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Index or (Cluster Size)', fontsize=10)
        ax.set_ylabel('Distance', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('hierarchical_linkage_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print cophenetic correlation results
    print("Cophenetic Correlation Coefficients:")
    print("=====================================")
    sorted_methods = sorted(cophenetic_scores.items(), key=lambda x: x[1], reverse=True)
    for method, score in sorted_methods:
        print(f"{method.capitalize():12s}: {score:.4f}")
    
    best_method = sorted_methods[0][0]
    print(f"\nBest linkage method: {best_method.capitalize()} (highest cophenetic correlation)")
    
    # Silhouette analysis for the best linkage method
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    inertias = []
    
    for k in n_clusters_range:
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage=best_method)
        cluster_labels = hierarchical.fit_predict(X_scaled)
        
        if len(np.unique(cluster_labels)) > 1:
            sil_score = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
    
    # Plot parameter selection results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cophenetic correlation comparison
    methods = list(cophenetic_scores.keys())
    scores = list(cophenetic_scores.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax1.bar(methods, scores, color=colors, alpha=0.8)
    ax1.set_title('Linkage Methods Comparison\n(Cophenetic Correlation)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cophenetic Correlation Coefficient', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Linkage Method', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight the best method
    best_idx = methods.index(best_method)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    # Silhouette analysis for optimal k
    ax2.plot(n_clusters_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax2.set_title(f'Optimal Clusters Analysis\n({best_method.capitalize()} Linkage)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Silhouette Score', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=6, color='red', linestyle='--', linewidth=2, label='k=6 (chosen)')
    ax2.legend(fontsize=12)
    
    # Add annotation for k=6
    k6_idx = 6 - min(n_clusters_range)
    if k6_idx < len(silhouette_scores):
        ax2.annotate(f'k=6\nScore: {silhouette_scores[k6_idx]:.3f}', 
                    xy=(6, silhouette_scores[k6_idx]), 
                    xytext=(8, silhouette_scores[k6_idx] + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold', color='red',
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('hierarchical_parameter_selection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_method, cophenetic_scores

def export_clustering_results_to_csv(data, clustering_results):
    """Export detailed clustering results with longitude, latitude, country names and cluster labels"""
    
    # 创建详细结果数据框，包含经度、纬度、国家名称和聚类标签
    detailed_results = []
    
    for method_name in ['K-means', 'Hierarchical']:
        predicted_labels = clustering_results[method_name]['labels']
        
        # 创建包含所有信息的结果数据框
        result_df = data[['longitude', 'latitude', 'country_name', 'continent']].copy()
        result_df['clustering_method'] = method_name
        result_df['cluster_label'] = predicted_labels
        
        # 重命名列为英文
        result_df = result_df.rename(columns={
            'longitude': 'Longitude',
            'latitude': 'Latitude', 
            'country_name': 'Country_Name',
            'continent': 'True_Continent',
            'clustering_method': 'Clustering_Method',
            'cluster_label': 'Cluster_Label'
        })
        
        detailed_results.append(result_df)
        
        print(f"\n=== {method_name} Detailed Results ===")
        print(f"Number of countries: {len(result_df)}")
        print("Sample data:")
        print(result_df.head())
    
    # 合并两个方法的详细结果
    combined_detailed_results = pd.concat(detailed_results, ignore_index=True)
    
    # 保存详细结果
    detailed_filename = 'clustering_detailed_results_combined.csv'
    combined_detailed_results.to_csv(detailed_filename, index=False, encoding='utf-8')
    
    print(f"\n=== DETAILED RESULTS EXPORTED ===")
    print(f"Detailed clustering results saved to: {detailed_filename}")
    print(f"Total records: {len(combined_detailed_results)}")
    print("\nColumns included:")
    print("- Longitude: 经度")
    print("- Latitude: 纬度") 
    print("- Country_Name: 国家名称")
    print("- True_Continent: 真实大洲")
    print("- Clustering_Method: 聚类方法")
    print("- Cluster_Label: 聚类标签")
    
    # 同时保留原来的聚类簇-大洲对应矩阵功能
    print("\n=== Creating Cluster-Continent Matrix ===")
    matrix_results = []
    
    for method_name in ['K-means', 'Hierarchical']:
        predicted_labels = clustering_results[method_name]['labels']
        
        # 创建结果数据框
        result_df = data.copy()
        result_df['predicted_cluster'] = predicted_labels
        
        # 创建聚类簇与大洲对应的统计表
        continent_cluster_matrix = pd.crosstab(
            result_df['continent'], 
            result_df['predicted_cluster'],
            margins=False  # 不显示总计行列
        )
        
        # 重置索引，将continent作为列
        continent_cluster_matrix_reset = continent_cluster_matrix.reset_index()
        
        # 添加方法名称列
        continent_cluster_matrix_reset.insert(0, 'Method', method_name)
        
        # 重命名聚类列
        cluster_columns = [col for col in continent_cluster_matrix_reset.columns if isinstance(col, int)]
        column_rename = {col: f'Cluster_{col}' for col in cluster_columns}
        continent_cluster_matrix_reset = continent_cluster_matrix_reset.rename(columns=column_rename)
        
        # 重命名continent列为英文
        continent_cluster_matrix_reset = continent_cluster_matrix_reset.rename(columns={'continent': 'Continent'})
        
        matrix_results.append(continent_cluster_matrix_reset)
        
        print(f"\n{method_name} Cluster-Continent Distribution Matrix:")
        print(continent_cluster_matrix)
    
    # 合并矩阵结果
    final_matrix_results = pd.concat(matrix_results, ignore_index=True)
    
    # 保存矩阵结果
    matrix_filename = 'clustering_continent_matrix_combined.csv'
    final_matrix_results.to_csv(matrix_filename, index=False, encoding='utf-8')
    
    print(f"\nCluster-continent matrix saved to: {matrix_filename}")
    
    return combined_detailed_results, final_matrix_results

def main():
    print("=== Countries Clustering Analysis ===\n")
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    main_data, antarctica_data = load_and_preprocess_data()
    
    # Analyze optimal number of clusters
    print("\n2. Analyzing optimal number of clusters...")
    scaler = optimal_clusters_analysis(main_data)
    
    # Hierarchical clustering parameter selection
    print("\n3. Hierarchical clustering parameter selection...")
    best_linkage, cophenetic_scores = hierarchical_parameter_selection(main_data)
    
    # Perform clustering with k=6 
    print(f"\n4. Performing clustering analysis with k=6...")
    print(f"   Using best linkage method: {best_linkage.capitalize()}")
    clustering_results, _, true_labels, continents = perform_clustering(main_data, n_clusters=6)
    
    # Store accuracies for comparison
    accuracies_dict = {}
    
    # Create visualizations for both methods
    print("\n5. Creating visualizations...")
    
    for method_name in ['K-means', 'Hierarchical']:
        print(f"\n--- {method_name} Analysis ---")
        
        # World map with continent coloring
        create_world_map_with_continents(main_data, antarctica_data, clustering_results, method_name)
        
        # Confusion matrix
        accuracies = plot_confusion_matrix(main_data, clustering_results, method_name, true_labels, continents)
        accuracies_dict[method_name] = accuracies
        
        print(f"\n{method_name} - Continent-wise Clustering Accuracy:")
        for continent, accuracy in accuracies.items():
            print(f"  {continent}: {accuracy:.1f}%")
    
    # Create accuracy comparison
    print("\n6. Creating accuracy comparison...")
    plot_accuracy_comparison(accuracies_dict)
    
    # Export clustering results to CSV
    print("\n7. Exporting clustering results to CSV...")
    detailed_results, matrix_results = export_clustering_results_to_csv(main_data, clustering_results)
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Total countries analyzed: {len(main_data)}")
    print(f"Antarctica (Outlier): {len(antarctica_data)}")
    print(f"Continents: {len(continents)}")
    
    print(f"\nHierarchical clustering parameter selection:")
    print(f"  Best linkage method: {best_linkage.capitalize()}")
    print(f"  Cophenetic correlation scores:")
    for method, score in sorted(cophenetic_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {method.capitalize():12s}: {score:.4f}")
    
    print(f"\nBest performing method:")
    best_method = max(accuracies_dict.keys(), 
                     key=lambda x: np.mean(list(accuracies_dict[x].values())))
    best_avg = np.mean(list(accuracies_dict[best_method].values()))
    print(f"  {best_method}: {best_avg:.1f}% average accuracy")
    
    print(f"\nGenerated files:")
    print(f"- optimal_clusters_analysis.png")
    print(f"- hierarchical_linkage_comparison.png")
    print(f"- hierarchical_parameter_selection.png")
    print(f"- clustering_accuracy_comparison.png")
    for method in ['K-means', 'Hierarchical']:
        print(f"- world_map_{method}.html/.png")
        print(f"- confusion_matrix_{method}.png")
    print(f"- clustering_detailed_results_combined.csv (详细结果：经度、纬度、国家名称、聚类标签)")
    print(f"- clustering_continent_matrix_combined.csv (聚类簇-大洲对应矩阵)")

if __name__ == "__main__":
    main()